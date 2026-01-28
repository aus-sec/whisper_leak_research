import importlib
import json
import math
from core.classifiers.lightgbm_classifier import LightGBMClassifier
from core.classifiers.loader import Loader
from core.utils import PrintUtils

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import random
import os
import torch.nn.functional as F

class BaseClassifier(nn.Module):
    """
        Base classifier class that can be extended by different model architectures.
    """

    def __init__(self, norm):
        """
            Creates an instance.
        """

        # Save members
        super().__init__()
        self.norm = norm
        self.max_len = norm['max_len']
        self.class_name = self.__class__.__name__
        self.args = {}

    def forward(self, x):
        """
            Implements a forward pass.
        """

        # Not implemented
        raise NotImplementedError('Subclasses must implement forward method')
    

    def save(self, filepath):
        """
            Save model state dictionary and normalization parameters.
        """

        # Saves a model
        torch.save(self.state_dict(), filepath)
        PrintUtils.print_extra(f'Model saved to file *{os.path.basename(filepath)}*')
        
        # Save normalization parameters
        if self.norm:
            norm_filepath = filepath.replace('.pth', '_norm_params.json')

            #with open(norm_filepath, 'w', encoding='utf-8') as f:
            #    json.dump({
            #        'normalization_params': self.norm,
            #        'class_name': self.class_name,
            #        'args': self.args,
            #    }, f, indent=4)
            
            PrintUtils.print_extra(f'Normalization parameters saved to {os.path.basename(norm_filepath)}')
    
    @classmethod
    def load(cls, filepath, device):
        """
            Returns a classifier instance loaded from a file.
        """

        # Load normalization parameters
        norm_filepath = filepath.replace('.pth', '_norm_params.json')
        if not os.path.exists(norm_filepath):
            raise Exception(f'Normalization parameters file not found: {norm_filepath}')
        
        loaded = json.load(open(norm_filepath, 'r', encoding='utf-8'))
        normalization_params = loaded['normalization_params']
        class_name = loaded['class_name']
        args = loaded['args']

        module_name = f"core.classifiers.{class_name.lower()}"

        if class_name == "AttentionBiLSTMClassifier":
            module_name = "core.classifiers.attention_bi_lstm_classifier"
        elif class_name == "CNNClassifier":
            module_name = "core.classifiers.cnn_classifier"
        elif class_name == "LSTMTransformerClassifier":
            module_name = "core.classifiers.lstm_transformer_classifier"
        elif class_name == "LightGBMClassifier":
            module_name = "core.classifiers.lightgbm_classifier"
        else:
            raise Exception(f'Unknown classifier type: {class_name}')
        
        module = importlib.import_module(module_name) # Needed to avoid circular import
        ClassifierClass = getattr(module, class_name)

        if class_name == "LightGBMClassifier":
            classifier = LightGBMClassifier.load(filepath)
        else:
            classifier = ClassifierClass(norm=normalization_params, **args)
            classifier.to(device)
            classifier.load_state_dict(torch.load(filepath, map_location=device))
            classifier.eval()
        PrintUtils.print_extra(f'Classifier loaded from file *{os.path.basename(filepath)}*')
        return classifier
    

    def inference(self, input_data, device):
        """
            Runs inference on the input data.
            The input data can be a DataFrame or a tuple of (time_diffs, data_lengths).
            Returns the predicted probabilities and binary predictions.
        """

        # Evaluate
        self.eval()
        
        # Handle DataFrames
        if isinstance(input_data, pd.DataFrame):
            
            # If DataFrame is provided, normalize it and create a dataset
            if self.norm is None:
                raise Exception('Normalization parameters must be provided for a DataFrame input')
            
            # Create a dataset
            dataset = Loader(input_data)
            dataset.apply_normalization(self.norm)
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
           
            # Get all probabilities
            all_probs = []
            with torch.no_grad():
                for X, _ in loader:
                    X = X.to(device)
                    output = self(X)
                    # Convert to probabilities
                    output = torch.sigmoid(output).cpu().numpy()
                    all_probs.extend(output.flatten())
            
            # Get the prediction based on the probabilities and return them
            all_probs = np.array(all_probs)
            predictions = (all_probs > 0.5).astype(int)
            return all_probs, predictions
        
        # Handle a 2-tuple
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            
            # If tuple of arrays is provided, normalize directly
            time_diffs, data_lengths = input_data
            if self.norm is None:
                raise Exception('Normalization parameters must be provided for raw input')
            #time_mean, time_std, size_mean, size_std, max_len = self.norm
            time_mean = self.norm['time_mean']
            time_std = self.norm['time_std']
            size_mean = self.norm['size_mean']
            size_std = self.norm['size_std']
            max_len = self.norm['max_len']

            normalized_time = []
            for val in time_diffs[:max_len]:  # Trim to max_len
                normalized_time.append((val - time_mean) / time_std)

            normalized_size = []
            for val in data_lengths[:max_len]:
                normalized_size.append((val - size_mean) / size_std)
            
            # Pad sequences
            time_padded = np.zeros(max_len)
            size_padded = np.zeros(max_len)
            time_padded[:len(normalized_time)] = normalized_time[:max_len]
            size_padded[:len(normalized_size)] = normalized_size[:max_len]
            
            # Prepare tensor
            sample = np.stack([time_padded, size_padded], axis=0)
            tensor_input = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                output = self(tensor_input)
                prob = torch.sigmoid(output).cpu().numpy().flatten()[0]
                prediction = 1 if prob > 0.5 else 0
            
            # Return the probability and prediction
            return prob, prediction

        # Unsupported format
        else:
            raise Exception(f'Input must be either a DataFrame or a tuple of (time_diffs, data_lengths)')
    
    