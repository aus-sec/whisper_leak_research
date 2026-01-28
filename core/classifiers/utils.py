import json
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from core.classifiers.lightgbm_classifier import LightGBMClassifier
from core.utils import PrintUtils, PromptUtils # Assuming this import works in your environment
import numpy as np
from sklearn.metrics import accuracy_score
import random
import sys
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ModelTrainer:
    """
    Handles model training, evaluation, and prediction across different model types.
    Provides a unified interface for working with various model architectures.
    """
    def __init__(self, model, config, device):
        """
        Initialize the model trainer.
        
        Args:
            model: The model to train
            config: The benchmark configuration
            device: The computation device (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.device = device
        self.history = None
        self.criterion = nn.BCEWithLogitsLoss() if isinstance(model, nn.Module) else None
        self.early_stopping = None
        
    def fit(self, train_data, val_data, save_path=None, batch_size=32):
        """
        Train the model with the provided data.
        
        Args:
            train_data: Training data (Loader)
            val_data: Validation data (Loader)
            save_path: Path to save the best model
            
        Returns:
            dict: Training history
        """
        if isinstance(self.model, LightGBMClassifier):
            # Handle LightGBM models (using DataFrames)
            self.model.fit(
                train_df=train_data.df,
                val_df=val_data.df,
                patience=self.config.patience
            )
            # Create basic history for consistency
            self.history = {
                'best_epoch': getattr(self.model.model, 'best_iteration_', 0),
                'train_losses': [],
                'val_losses': [],
                'train_accs': [],
                'val_accs': [],
            }
        
        elif isinstance(self.model, nn.Module):
            # Create a loader
            train_loader = DataLoader(train_data, batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size, shuffle=False)

            # Handle PyTorch models (using DataLoaders)
            self.model = self.model.to(self.device)
            
            # Setup optimizer (handle model-specific param groups)
            if hasattr(self.model, 'get_optimizer_params'):
                optimizer_params = self.model.get_optimizer_params(self.config.learning_rate)
                PrintUtils.print_extra(f"Using model-specific parameter groups for optimizer")
                optimizer = optim.Adam(optimizer_params)
            else:
                optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            
            # Setup early stopping
            early_stopping_path = save_path.replace('.pth', '_best.pt') if save_path else 'best_model.pt'
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                verbose=True,
                path=early_stopping_path
            )
            
            # Training loop
            self.history = self._train_pytorch_model(train_loader, val_loader, optimizer)
        
        else:
            raise TypeError(f"Model type {type(self.model)} is not supported for training.")
        
        # Save the final model if path provided
        if save_path:
            self.model.save(save_path)
            
        return self.history
    
    def _train_pytorch_model(self, train_loader, val_loader, optimizer):
        """
        Inner training loop for PyTorch models.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: The optimizer to use
            
        Returns:
            dict: Training history
        """
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_epoch = 0
        
        for epoch in range(self.config.max_epochs):
            PrintUtils.start_stage(f'Training (epoch {epoch+1}/{self.config.max_epochs})')

            # Training phase
            self.model.train()
            train_loss, train_acc = 0.0, 0.0
            steps = 0
            
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    train_acc += (preds == y).sum().item() / y.size(0)
                
                steps += 1
                
                # Update progress
                progress = (batch_idx + 1) / len(train_loader)
                PrintUtils.start_stage(
                    f'Training (epoch {epoch+1}/{self.config.max_epochs}): {progress*100:.1f}%', 
                    override_prev=True
                )
            
            # Calculate epoch metrics
            train_loss /= steps
            train_acc /= steps
            
            # Validation phase
            self.model.eval()
            val_loss, val_acc = 0.0, 0.0
            steps = 0
            
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device).float().unsqueeze(1)
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_acc += (preds == y).sum().item() / y.size(0)
                    
                    steps += 1
            
            val_loss /= steps
            val_acc /= steps
            
            # Store history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Early stopping check
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                PrintUtils.print_extra(f"Early stopping triggered at epoch {epoch+1}")
                best_epoch = epoch + 1 - self.early_stopping.counter
                break
            
            # Log progress
            PrintUtils.start_stage(
                f'Epoch {epoch+1}/{self.config.max_epochs} - train loss: {train_loss:.4f}, '
                f'train acc: {train_acc:.4f} - val loss: {val_loss:.4f}, val acc: {val_acc:.4f}',
                override_prev=True
            )
            PrintUtils.end_stage()
            
            # Update best epoch if no early stopping
            if epoch == self.config.max_epochs - 1:
                best_epoch = self.early_stopping.counter
        
        # Load best model
        if self.early_stopping and self.early_stopping.path:
            self.model.load_state_dict(torch.load(self.early_stopping.path))
        
        PrintUtils.end_stage()
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_epoch': best_epoch
        }
    
    def predict(self, data, batch_size=32):
        """
        Make predictions with the trained model.
        
        Args:
            data: Test data (DataLoader or DataFrame)
            
        Returns:
            tuple: (scores, labels, loss)
        """
        if isinstance(self.model, LightGBMClassifier):
            # For LightGBM, data should be a DataFrame
            data = data.df
            
            # Get predictions
            try:
                scores = self.model.predict_proba(data)
                labels = data['target'].values
                return scores, labels, None  # Loss not available for LightGBM
            except Exception as e:
                PrintUtils.print_extra(f"Error during LightGBM prediction: {e}")
                return None, None, None
        
        elif isinstance(self.model, nn.Module):
            # For PyTorch models, data should be a DataLoader
            data = DataLoader(data, batch_size, shuffle=False)
            
            self.model.eval()
            all_scores = []
            all_labels = []
            total_loss = 0.0
            
            with torch.no_grad():
                for X, y in data:
                    X = X.to(self.device)
                    y_device = y.to(self.device).float().unsqueeze(1)
                    outputs = self.model(X)
                    
                    if self.criterion:
                        loss = self.criterion(outputs, y_device)
                        total_loss += loss.item() * X.size(0)
                    
                    scores = torch.sigmoid(outputs)
                    all_scores.extend(scores.cpu().numpy().flatten())
                    all_labels.extend(y.numpy().flatten())
            
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
            
            epoch_loss = total_loss / len(data.dataset) if self.criterion and len(data.dataset) > 0 else 0
            return all_scores, all_labels, epoch_loss
        
        else:
            raise TypeError(f"Model type {type(self.model)} is not supported for prediction.")



class EarlyStopping(object):
    """
        Early stopping to prevent overfitting.
    """

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
            Creates an instance.
            The patience is an integer that specifies how many epochs to wait after last improvement.
            The delta is a floating point number that containsthe minimum change to qualify as an improvement.
        """

        # Save members
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        """
            Call override.
        """

        # Performs the early stopping
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            PrintUtils.print_extra(f'EarlyStopping counter: *{self.counter}* out of *{self.patience}*')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """
            Saves model when validation accuracy improves.
        """
        
        # Save data
        if self.verbose:
            PrintUtils.print_extra(f'Validation loss improved (*{self.val_loss_min:.6f}* --> *{val_loss:.6f}*). Saving model.')
        torch.save(model.state_dict(), self.path)

        # Override accuracy value 
        self.val_loss_min = val_loss


def set_seed(seed=42):
    """
        Set the random seed for reproducibility.
    """

    # Set the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    PrintUtils.print_extra(f'Random seed set to *{seed}*')

import time
# import numpy as np # Assuming numpy is available if needed for other parts
# from utils import PrintUtils # Assuming PrintUtils is available if needed for other parts
import sys # Needed for sys.exit()
from collections import defaultdict # Useful for storing timing data

# Define NUM_ITERATIONS_TO_TIME constant
NUM_ITERATIONS_TO_TIME = 20


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, max_epochs):
    """
        Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    seconds_per_steps = []
    losses_per_steps = []
    accuracies_per_steps = []

    for i, (X, y) in enumerate(dataloader): # Use enumerate for progress tracking
        start_time_epoch = time.time()
        X, y = X.to(device), y.to(device).float().unsqueeze(1) # Ensure y is float and correct shape

        optimizer.zero_grad()
        output = model(X) # Output should be logits if using BCEWithLogitsLoss

        loss = criterion(output, y) # Calculate loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

        # Calculate predictions based on whether output is logits or probabilities
        with torch.no_grad():
            # If criterion expects logits, output is logits. Apply sigmoid for prediction.
            pred = (torch.sigmoid(output) > 0.5).float()

        correct += (pred == y).sum().item()
        total += y.size(0)

        current_batch_loss = loss.item()
        current_batch_accuracy = (pred == y).sum().item() / y.size(0) if y.size(0) > 0 else 0

        # Update progress bar
        seconds_per_step = (time.time() - start_time_epoch)
        seconds_per_steps.append(seconds_per_step)
        losses_per_steps.append(current_batch_loss)
        accuracies_per_steps.append(current_batch_accuracy)
        # Use a moving average window (e.g., last 20 steps) for smoother estimates
        if len(seconds_per_steps) > 20: 
             seconds_per_steps.pop(0) 
             losses_per_steps.pop(0)
             accuracies_per_steps.pop(0)
        avg_seconds_per_step = np.mean(seconds_per_steps) if seconds_per_steps else 0
        avg_loss_per_step = np.mean(losses_per_steps) if losses_per_steps else 0
        avg_accuracy_per_step = np.mean(accuracies_per_steps) if accuracies_per_steps else 0
        
        progress = (i + 1) / len(dataloader)
        PrintUtils.start_stage(
            f'Training (epoch {epoch+1}/{max_epochs}): {progress*100:.1f}% '
            f'(s/iter={avg_seconds_per_step:.3f}, loss={avg_loss_per_step:.4f}, acc={avg_accuracy_per_step:.3f})', 
            override_prev=True
        )

    # Return epoch loss and accuracy
    epoch_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


def get_prediction_scores(model, dataloader_or_df, device, criterion=None, return_probs=True, neg_to_pos_ratio=None):
    """
        Get raw prediction scores (logits or probabilities), true labels and losses from dataloader.

        Args:
            model: The trained model.
            dataloader: DataLoader for the dataset.
            device: The device to run inference on.
            criterion: Loss function to calculate per-batch loss. If None, losses won't be computed.
            return_probs (bool): If True, applies sigmoid to model output assuming 
                                 output are logits, returning probabilities. 
                                 If False, returns raw model output (logits).
                                 Set to False only if downstream code explicitly handles logits.
            neg_to_pos_ratio: Ratio of negative to positive samples (for imbalanced datasets).

        Returns:
            tuple (np.array, np.array, float): A tuple containing scores, true labels, and total loss (if criterion provided).
    """
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0.0

    if isinstance(model, LightGBMClassifier):
        PrintUtils.print_extra("Getting predictions using LightGBMClassifier predict_proba")
        if dataloader_or_df is None:
             PrintUtils.print_extra("Error: dataloader_or_df is required for LightGBMClassifier prediction.")
             return None, None, None
        if 'target' not in dataloader_or_df.columns:
            PrintUtils.print_extra("Error: dataloader_or_df must contain 'target' column for evaluation.")
            return None, None, None

        try:
            # Note: LightGBM predict_proba uses the internal _prepare_features
            # which expects the original 'time_diffs', 'data_lengths' columns.
            # Ensure dataloader_or_df has these columns in the correct format.
             all_scores = model.predict_proba(dataloader_or_df)
             all_labels = dataloader_or_df['target'].values
             loss = np.nan # LightGBM doesn't compute loss in the same way during predict
             PrintUtils.print_extra(f"Generated {len(all_scores)} predictions from LightGBM.")
             return np.array(all_scores), np.array(all_labels), loss
        except Exception as e:
             PrintUtils.print_extra(f"Error during LightGBM prediction: {e}")
             import traceback
             traceback.print_exc()
             return None, None, None
    
    else:
        with torch.no_grad():
            for X, y in dataloader_or_df:
                X = X.to(device)
                y_device = y.to(device).float().unsqueeze(1) if criterion else y  # Only move to device if needed
                outputs = model(X)  # Raw model output (likely logits)

                # Calculate loss if criterion is provided
                if criterion:
                    loss = criterion(outputs, y_device)
                    total_loss += loss.item() * X.size(0)
                
                if return_probs:
                    # Assume outputs are logits if return_probs is True, apply sigmoid
                    scores = torch.sigmoid(outputs) 
                else:
                    # Return raw logits
                    scores = outputs 

                all_scores.extend(scores.cpu().numpy().flatten())
                all_labels.extend(y.numpy().flatten())  # y comes from dataloader, usually on CPU already
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        epoch_loss = total_loss / len(dataloader_or_df.dataset) if criterion and len(dataloader_or_df.dataset) > 0 else 0
        return all_scores, all_labels, epoch_loss


def eval_epoch(model, dataloader, criterion, device, neg_to_pos_ratio=None):
    """
        Evaluate the model on the validation set.
    """
    model.eval()
    
    # Use get_prediction_scores to get predictions and loss
    scores, labels, epoch_loss = get_prediction_scores(
        model, 
        dataloader, 
        device, 
        criterion=criterion,
        return_probs=True,  # If using BCEWithLogitsLoss, we want probabilities
        neg_to_pos_ratio=neg_to_pos_ratio
    )
    
    # Convert scores to binary predictions
    predictions = (scores > 0.5).astype(float)
    
    # Calculate accuracy using all collected predictions and labels
    accuracy = accuracy_score(labels, predictions) if len(labels) > 0 else 0
    
    return epoch_loss, accuracy


def split_data(df, seed, test_size=0.2, valid_size=0.1):
    """
    Split data into train, validation, and test sets
    
    Args:
        df: DataFrame to split
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Split into train and test sets preserving prompt distribution
    unique_prompts = df.drop_duplicates(subset=['prompt'])[['prompt', 'target']]
    train_and_val_prompts, test_prompts = train_test_split(
        unique_prompts['prompt'],
        test_size=test_size,
        random_state=seed,  # Use trial-specific seed
        stratify=unique_prompts['target']
    )
    test_prompts = set(test_prompts)
    
    # Split train into train and validation
    df_train_val = df[df['prompt'].isin(train_and_val_prompts)]
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=valid_size,
        random_state=seed,  # Use trial-specific seed
        stratify=df_train_val['target']
    )
    
    df_test = df[df['prompt'].isin(test_prompts)]
    
    return df_train, df_val, df_test

def calculate_sampling_details(df, neg_to_pos):
    """
    Calculates the number of positive and negative samples, the desired
    number of negative samples based on the ratio, and the current ratio.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'target' column.
        neg_to_pos (float): The desired ratio of negative to positive samples.

    Returns:
        tuple: Contains:
            - n_pos (int): Count of positive samples (target=1).
            - n_neg (int): Count of negative samples (target=0).
            - n_neg_desired (int): Target count for negative samples based on ratio.
                                    Returns current n_neg if n_pos is 0.
            - current_ratio (float): Current neg/pos ratio (n_neg / n_pos).
                                     Returns np.inf if n_pos=0 and n_neg>0,
                                     np.nan if n_pos=0 and n_neg=0.
    Raises:
        ValueError: If 'target' column is missing or neg_to_pos is not positive.
    """
    if 'target' not in df.columns:
        raise ValueError("DataFrame must have a 'target' column.")
    if not isinstance(neg_to_pos, (int, float)) or neg_to_pos <= 0:
        raise ValueError("neg_to_pos must be a positive number.")
    if not df['target'].isin([0, 1]).all():
         print("Warning: 'target' column contains values other than 0 and 1. Assuming 0=negative, 1=positive.")

    n_pos = df['target'].eq(1).sum()
    n_neg = df['target'].eq(0).sum()

    if n_pos == 0:
        n_neg_desired = n_neg # Cannot determine target based on ratio
        current_ratio = np.inf if n_neg > 0 else np.nan
    else:
        # Use round to get the closest integer count for the target ratio.
        n_neg_desired = int(round(neg_to_pos * n_pos))
        current_ratio = n_neg / n_pos

    return n_pos, n_neg, n_neg_desired, current_ratio


def load_chatbot_data(chatbot_name, input_folder, prompts_file, downsample_rate=1.0):
    """
    Load and preprocess data for the specified chatbot.
    
    Args:
        chatbot_name: Name of the chatbot
        input_folder: Folder containing data files
        prompts_file: Path to prompts JSON file
        downsample_rate: Fraction of files to load (0.0-1.0)
        
    Returns:
        DataFrame: Processed data
    """
    PrintUtils.start_stage('Loading sequences data')
    training_set_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        input_folder
    )
    aggregated_files = [
        os.path.join(training_set_dir, filename)
        for filename in os.listdir(training_set_dir)
        if filename.lower() == f'{chatbot_name.lower()}.json'
            or (
                filename.lower().startswith(f'{chatbot_name.lower()}_')
                and filename.lower().endswith('.json')
            )
    ]

    if aggregated_files:
        files = aggregated_files
    else:
        files = [
            os.path.join(training_set_dir, filename)
            for filename in os.listdir(training_set_dir)
            if filename.lower().endswith(f'_{chatbot_name.lower()}.seq')
                or f'_{chatbot_name.lower()}_' in filename.lower()
                or filename.lower() == f'{chatbot_name.lower()}.seq'
        ]
    
    if not files:
        raise ValueError(f'Did not find training set files for chatbot {chatbot_name}')
    
    if downsample_rate < 1.0 and len(files) > 1:
        # Downsample the files to a fraction of the original size
        files = files[:int(len(files) * downsample_rate)]

    data = []
    for file_index, file_path in enumerate(files):
        with open(file_path, 'r', encoding="utf8") as fp:
            new_data = json.load(fp)
            # Check if it's a list of dictionaries
            if isinstance(new_data, list):
                # Randomly sample
                if downsample_rate < 1.0:
                    new_data = random.sample(new_data, int(len(new_data) * downsample_rate))

                # Extend the data list with new data
                data.extend(new_data)
            else:
                data.append(json.load(fp))
        
        if file_index % 10 == 0:
            percentage = (file_index * 100) // len(files)
            PrintUtils.start_stage(
                f'Loading sequences data ({file_index} / {len(files)} = {percentage}%)', 
                override_prev=True
            )
    
    df = pd.DataFrame(data)
    PrintUtils.end_stage()

    # Join to prompts to add target column
    prompts = PromptUtils.read_prompts(prompts_file)
    df['target'] = df['prompt'].apply(lambda x: 1 if x in prompts['positive']['prompts'] else 0)
    
    total_prompts = len(prompts['positive']['prompts']) + len(prompts['negative']['prompts'])
    PrintUtils.print_extra(f'Loaded {total_prompts} prompts')
    PrintUtils.print_extra(f'Loaded {len(df)} samples for {chatbot_name}')
    
    return df
