#!/usr/bin/env python3
"""
Whisper Leak Benchmark - A tool for benchmarking ML models against multiple chatbots
to detect prompt leakage, with support for data mitigations.
"""
from pathlib import Path
import yaml
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from enum import Enum
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast # Added for literal_eval

from core.chatbot_utils import ChatbotUtils
from core.classifiers.attention_bi_lstm_classifier import AttentionBiLSTMClassifier
from core.classifiers.bert_time_series_classifier import BERTTimeSeriesClassifier
from core.classifiers.cnn_classifier import CNNClassifier
from core.classifiers.combined_lstm_bert_classifier import CombinedLSTMBERTClassifier
from core.classifiers.lightgbm_classifier import LightGBMClassifier
from core.classifiers.lstm_transformer_classifier import LSTMTransformerClassifier
from core.classifiers.loader import Loader

# Import Mitigation framework
from core.classifiers.mitigations import apply_mitigation_set, BaseMitigation

from core.classifiers.utils import (
    EarlyStopping, ModelTrainer, load_chatbot_data,
    set_seed, split_data
)

from core.classifiers.visualization import (
    calculate_metrics, set_plot_style, plot_training_curves, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix,
    plot_score_distribution, create_model_dashboard
)

from core.utils import ThrowingArgparse, PrintUtils

import json



class FeatureMode(Enum):
    """
    Enum representing the feature mode used for classification
    """
    BOTH = "Both"
    DATA_SIZE_ONLY = "Data Size Only"
    TIME_ONLY = "Time Only"

    # Allow initialization from string
    @classmethod
    def from_string(cls, s: str):
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"'{s}' is not a valid FeatureMode")


class BenchmarkConfig:
    """
    Configuration class for the benchmark script, loaded from YAML.
    """
    def __init__(self, config_path):
        """
        Initialize the configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file
        """
        PrintUtils.start_stage(f'Loading configuration from {config_path}')
        try:
            with open(config_path, 'r', encoding='utf8') as f:
                self.config = yaml.safe_load(f)

            # --- Basic Setup ---
            self.benchmark_name = self.config.get('benchmark_name', 'benchmark')
            self.num_trials = self.config.get('num_trials', 1)

            # --- Data Path ---
            self.data_path = self.config.get('data_path', 'data_v2')
            if isinstance(self.data_path, str):
                self.data_path = [self.data_path]
            elif not isinstance(self.data_path, list):
                 raise ValueError("data_path must be a string or a list of strings")

            # --- Chatbots ---
            self.chatbots = self.config.get('chatbots', [])
            if not self.chatbots:
                raise ValueError("No chatbots specified in configuration")
            if '*' in self.chatbots:
                # Load all available chatbots if '*' is present
                all_bots_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'chatbots'
                )
                if not os.path.isdir(all_bots_path):
                     raise FileNotFoundError(f"Chatbot directory not found: {all_bots_path}")
                self.chatbots = list(ChatbotUtils.load_chatbots(all_bots_path).keys())
                # Sort for predictable order before shuffling
                self.chatbots.sort()
                np.random.seed(42) # Use fixed seed for shuffling list
                np.random.shuffle(self.chatbots)
                print(f"Expanded chatbots to: {self.chatbots}")

            # --- Sampling Rates ---
            sampling_rate_config = self.config.get('sampling_rate', 1.0)
            if isinstance(sampling_rate_config, (float, int)):
                self.sampling_rates = [float(sampling_rate_config)]
            elif isinstance(sampling_rate_config, list):
                self.sampling_rates = sorted([float(rate) for rate in sampling_rate_config]) # Sort rates
            else:
                raise ValueError("sampling_rate must be a float or a list of floats")
            if not all(0.0 < rate <= 1.0 for rate in self.sampling_rates):
                 raise ValueError("All sampling_rates must be between 0.0 (exclusive) and 1.0 (inclusive)")

            # --- Feature Modes ---
            feature_mode_strings = self.config.get('feature_modes', [FeatureMode.BOTH.value, FeatureMode.DATA_SIZE_ONLY.value, FeatureMode.TIME_ONLY.value])
            if not isinstance(feature_mode_strings, list):
                raise ValueError("feature_modes must be a list of strings (e.g., ['Both', 'Data Size Only'])")
            self.feature_modes = [FeatureMode.from_string(fm) for fm in feature_mode_strings]


            # --- Mitigations ---
            self.mitigations = self.config.get('mitigations', {'none': []})
            if not isinstance(self.mitigations, dict):
                raise ValueError("mitigations must be a dictionary where keys are set names and values are lists of mitigation configs.")
            # Ensure 'none' mitigation set exists if not provided
            if 'none' not in self.mitigations:
                 self.mitigations['none'] = []
                 PrintUtils.print_warning("No 'none' mitigation set found. Adding an empty 'none' set.")


            # --- Model Config ---
            model_config = self.config.get('model', {})
            if not model_config:
                raise ValueError("No model parameters specified in configuration")
            self.model_class = model_config.get('class', 'AttentionBiLSTMClassifier') # Updated default
            self.model_params = model_config.get('params', {})

            # --- Training Parameters ---
            training_params = self.config.get('training', {})
            self.training_params = training_params
            self.batch_size = training_params.get('batch_size', 32)
            self.max_epochs = training_params.get('max_epochs', 100) # Reduced default
            self.learning_rate = training_params.get('learning_rate', 0.0002) # Adjusted default
            self.patience = training_params.get('patience', 10) # Adjusted default
            self.test_size = training_params.get('test_size', 20)
            self.valid_size = training_params.get('valid_size', 5)
            self.seed = training_params.get('seed', 42)

            # --- Print Summary ---
            PrintUtils.print_extra(f'Configuration loaded for model: *{self.model_class}*')
            PrintUtils.print_extra(f'Benchmark name: *{self.benchmark_name}*')
            PrintUtils.print_extra(f'Chatbots: *{", ".join(self.chatbots)}*')
            PrintUtils.print_extra(f'Trials: *{self.num_trials}*')
            PrintUtils.print_extra(f'Sampling Rates: *{[f"{rate*100:.1f}%" for rate in self.sampling_rates]}*')
            PrintUtils.print_extra(f'Feature Modes: *{", ".join([fm.value for fm in self.feature_modes])}*')
            PrintUtils.print_extra(f'Mitigation Sets: *{", ".join(self.mitigations.keys())}*')
            PrintUtils.print_extra(f'Data Paths: *{", ".join(self.data_path)}*')


        except Exception as e:
            PrintUtils.end_stage(fail_message=f"Failed to load configuration: {str(e)}")
            raise

        PrintUtils.end_stage()

class BenchmarkRunner:
    """
    Main class for running the benchmark suite, now supporting mitigations
    and multiple feature modes.
    """
    def __init__(self, config_path, reprocess_only=False):
        """
        Initialize the benchmark runner

        Args:
            config_path: Path to the YAML configuration file
            reprocess_only: If True, only reprocess statistics without retraining
        """
        self.config = BenchmarkConfig(config_path)
        self.reprocess_only = reprocess_only
        self.results = [] # Stores results dictionaries

        # --- Setup Directories ---
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.benchmark_dir = os.path.join(self.base_dir, 'benchmark_results') # Renamed dir
        os.makedirs(self.benchmark_dir, exist_ok=True)

        # Results CSV file path
        self.results_file = os.path.join(self.benchmark_dir, f'{self.config.benchmark_name}_results.csv')

        # Base output directory for plots, models, etc.
        self.output_dir = os.path.join(self.benchmark_dir, self.config.benchmark_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Setup Device ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        PrintUtils.print_extra(f'Using device: *{self.device}*')

        self.existing_results = set() # Track already processed runs

        # --- Set Seed & Style ---
        set_seed(self.config.seed) # Set global seed initially
        set_plot_style()

    def _get_run_key_from_entry(self, entry):
        """
        Generate a unique key for a specific run based on its parameters.

        Args:
            entry (dict): Dictionary containing the run parameters.

        Returns:
            tuple: A tuple representing the unique key for this run.
        """
        return (
            entry['CommonName'],
            entry['ChatBot'],
            entry['Features'],
            entry['Trial'],
            entry['Sampling Rate'],
            entry['Mitigation Set'],
            entry['Data Path']
        )

    def _get_run_key(self, chatbot, common_name, feature_mode, trial, sampling_rate, mitigation_set_name, data_path):
        """
        Generate a unique key for a specific run based on its parameters.

        Args:
            chatbot (str): Name of the chatbot.
            common_name (str): Common name (usually same as chatbot).
            feature_mode (FeatureMode): Feature mode enum.
            trial (int): Trial number.
            sampling_rate (float): Sampling rate.
            mitigation_set_name (str): Name of the mitigation set.
            data_path (str): Path to the data file.

        Returns:
            tuple: A tuple representing the unique key for this run.
        """
        return (
            common_name,
            chatbot,
            feature_mode.value, # Use enum value (string) for the key
            trial,
            sampling_rate,
            mitigation_set_name,
            data_path
        )

    def should_process_run(self, chatbot, common_name, feature_mode, trial, sampling_rate, mitigation_set_name, data_path):
        """
        Determine if we should process this specific combination based on existing results.

        Args:
            chatbot (str): Name of the chatbot.
            common_name (str): Common name (usually same as chatbot).
            feature_mode (FeatureMode): Feature mode enum.
            trial (int): Trial number.
            sampling_rate (float): Sampling rate.
            mitigation_set_name (str): Name of the mitigation set.
            data_path (str): Path to the data file.

        Returns:
            bool: True if the run should be processed, False otherwise.
        """
        key = self._get_run_key(
            chatbot,
            common_name,
            feature_mode,
            trial,
            sampling_rate,
            mitigation_set_name,
            data_path
        )
        return key not in self.existing_results

    def _get_run_output_dir(self, chatbot, mitigation_set_name, feature_mode, trial, data_path_basename):
        """Constructs the output directory path for a specific run."""
        # Use basename of data_path to avoid overly long paths if data_path includes slashes
        return os.path.join(
            self.output_dir,
            chatbot,
            mitigation_set_name,
            feature_mode.value.replace(" ", "_"), # Replace spaces for dir name
            f"data_{data_path_basename}",
            f"trial_{trial}"
        )

    def run_benchmark(self):
        """
        Run the full benchmark suite across all configured dimensions
        (chatbots, trials, feature modes, sampling rates, mitigation sets, data paths).
        """
        PrintUtils.start_stage(f'Starting benchmark suite: *{self.config.benchmark_name}*')

        # Load prompts file (needed for data loading)
        prompts_path = os.path.join(self.base_dir, 'prompts/standard/prompts.json')
        if not os.path.exists(prompts_path):
            PrintUtils.end_stage(fail_message=f"Prompts file not found at: {prompts_path}")
            return
        try:
            with open(prompts_path, 'r') as f:
                prompts = json.load(f) # Not directly used here, but good practice to load early
        except Exception as e:
             PrintUtils.end_stage(fail_message=f"Failed to load prompts file: {e}")
             return

        # Load existing results into memory if file exists
        if os.path.exists(self.results_file):
            try:
                existing_df = pd.read_csv(self.results_file)
                # Convert to list of dicts, handling potential NaN issues
                self.results = [row.to_dict() for _, row in existing_df.iterrows()]

                # Generate existing results set for quick lookup
                self.existing_results = set(
                    self._get_run_key_from_entry(entry) for entry in self.results
                )

                PrintUtils.print_extra(f'Loaded {len(self.results)} previous results from CSV into memory.')
            except Exception as e:
                PrintUtils.print_warning(f'Could not load existing results file {self.results_file} into memory: {str(e)}. Starting fresh results list.')
                self.results = []
        else:
            self.existing_results = set() # No existing results file, start fresh


        # --- Main Benchmark Loop ---
        total_runs = (
            self.config.num_trials *
            len(self.config.mitigations) *
            len(self.config.feature_modes) *
            len(self.config.sampling_rates) *
            len(self.config.data_path) *
            len(self.config.chatbots)
        )
        run_counter = 0

        PrintUtils.start_stage(f'Starting benchmark ({total_runs} total runs configured)')

        for trial in range(1, self.config.num_trials + 1):
          for mitigation_set_name, mitigation_configs in self.config.mitigations.items():
            for feature_mode in self.config.feature_modes:
              for current_sampling_rate in self.config.sampling_rates:
                for data_path in self.config.data_path:
                  for chatbot in self.config.chatbots:
                    run_counter += 1
                    data_path_basename = os.path.basename(data_path) # For output dir

                    # --- Check if run should be skipped ---
                    if self.should_process_run(chatbot, chatbot, feature_mode, trial, current_sampling_rate, mitigation_set_name, data_path) or self.reprocess_only:

                        # Print progress
                        PrintUtils.start_stage(
                             f'Run {run_counter}/{total_runs}: Bot={chatbot}, Mit={mitigation_set_name}, Feat={feature_mode.value}, Trial={trial}, Rate={current_sampling_rate*100:.0f}%, Data={data_path_basename}',
                             override_prev=True)

                        # --- Setup Output Directory for this Run ---
                        run_output_dir = self._get_run_output_dir(chatbot, mitigation_set_name, feature_mode, trial, data_path_basename)
                        os.makedirs(run_output_dir, exist_ok=True)

                        # --- Process or Reprocess ---
                        try:
                            if self.reprocess_only:
                                PrintUtils.print_extra("Reprocessing stats (not implemented)")
                                # self.reprocess_chatbot_statistics(...) # Placeholder
                            else:
                                # Process (Load, Mitigate, Train, Eval)
                                self.process_chatbot(
                                    chatbot=chatbot,
                                    common_name=chatbot, # Using chatbot name as common name
                                    data_path=data_path,
                                    output_dir=run_output_dir,
                                    prompts_file=prompts_path,
                                    feature_mode=feature_mode,
                                    trial=trial,
                                    sampling_rate=current_sampling_rate,
                                    mitigation_set_name=mitigation_set_name,
                                    mitigation_configs=mitigation_configs
                                )
                        except Exception as e:
                            PrintUtils.print_error(f"FAILED Run {run_counter}: Bot={chatbot}, Mit={mitigation_set_name}, Feat={feature_mode.value}, Trial={trial}. Error: {str(e)}")
                            import traceback
                            traceback.print_exc() # Print stack trace for debugging
                            # Continue to the next run

                    else:
                        # Skip message
                        PrintUtils.start_stage(
                             f'Run {run_counter}/{total_runs}: SKIPPING Bot={chatbot}, Mit={mitigation_set_name}, Feat={feature_mode.value}, Trial={trial}, Rate={current_sampling_rate*100:.0f}%, Data={data_path_basename} (already exists)',
                             override_prev=True)


        # --- Save Final Results ---
        PrintUtils.end_stage() # End overall benchmark stage
        if self.results:
            try:
                results_df = pd.DataFrame(self.results)
                # Reorder columns for better readability
                cols_order = [
                    'CommonName', 'ChatBot', 'Mitigation Set', 'Features', 'Trial',
                    'Sampling Rate', 'Data Path', 'AUPRC', 'F1 Score', 'Accuracy',
                    'Precision', 'Recall', 'Specificity', 'Best Epoch/Iter',
                    'Model Class', 'Train Size', 'Val Size', 'Test Size',
                    # Add other metrics if needed
                ]
                # Include only columns that actually exist in the dataframe
                final_cols = [col for col in cols_order if col in results_df.columns]
                # Add any remaining columns not in the preferred order
                final_cols.extend([col for col in results_df.columns if col not in final_cols])

                results_df = results_df[final_cols]
                results_df.to_csv(self.results_file, index=False, float_format='%.4f')
                PrintUtils.print_extra(f'Final results saved to *{self.results_file}*')
            except Exception as e:
                PrintUtils.print_error(f"Failed to save final results to CSV: {e}")
        else:
            PrintUtils.print_warning("No results were generated during the benchmark.")

        PrintUtils.print_extra(f'Benchmark suite "{self.config.benchmark_name}" finished.')


    def process_chatbot(self, chatbot, common_name, data_path, output_dir, prompts_file,
                        feature_mode: FeatureMode, trial: int, sampling_rate: float,
                        mitigation_set_name: str, mitigation_configs: list):
        """
        Process a single chatbot run: Load data, apply mitigations, split,
        train model, evaluate, and save results/artifacts.
        """
        run_desc = f"Bot={chatbot}, Mit={mitigation_set_name}, Feat={feature_mode.value}, Trial={trial}, Rate={sampling_rate*100:.0f}%"
        PrintUtils.start_stage(f'Processing: {run_desc}')

        try:
            # --- Set Trial Seed ---
            trial_seed = self.config.seed + trial
            set_seed(trial_seed)
            PrintUtils.print_extra(f'Using trial seed: *{trial_seed}*')

            # --- Load Data ---
            # Load full dataset initially
            df_full = load_chatbot_data(chatbot, data_path, prompts_file, downsample_rate=1.0) # Load all initially
            if df_full.empty:
                 raise ValueError(f"No data loaded for {chatbot} from {data_path}.")

            # --- Ensure Correct List Format BEFORE Mitigation ---
            # Mitigations expect lists, not strings
            if 'time_diffs' in df_full.columns:
                 df_full['time_diffs'] = df_full['time_diffs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            if 'data_lengths' in df_full.columns:
                 df_full['data_lengths'] = df_full['data_lengths'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            # --- Apply Mitigations ---
            # Calculate median total time
            def calculate_median_totals(df):
                median_total_time = df['time_diffs'].apply(lambda x: np.sum(x)).median()
                median_total_size = df['data_lengths'].apply(lambda x: np.sum(x)).median()
                # Calculate the median time deltas of all rows together ignoring time diffs that are < some small delta
                all_time_deltas = []
                for time_list in df['time_diffs']:
                    if isinstance(time_list, list):
                        for time_val in time_list:
                            if isinstance(time_val, (int, float)) and not np.isnan(time_val) and time_val > 0.005:
                                all_time_deltas.append(time_val)
                
                median_time_delta = np.median(all_time_deltas) if all_time_deltas else 0.0

                return median_total_time, median_total_size, median_time_delta

            # Apply mitigation set to the full dataframe before splitting
            median_time, median_size, median_time_delta = calculate_median_totals(df_full)
            df_mitigated = apply_mitigation_set(df_full, mitigation_configs)
            median_time_after_mitigations, median_size_after_mitigations, median_time_delta_after_mitigations = calculate_median_totals(df_mitigated)

            # --- Enforce feature mode by copying columns over ---
            if feature_mode == FeatureMode.DATA_SIZE_ONLY:
                df_mitigated["time_diffs"] = df_mitigated["data_lengths"].copy()
            elif feature_mode == FeatureMode.TIME_ONLY:
                df_mitigated["data_lengths"] = df_mitigated["time_diffs"].copy()

            del df_full # Free memory

            # --- Split Data ---
            # Split the *mitigated* data
            PrintUtils.start_stage('Splitting mitigated data')
            df_train_orig, df_val, df_test = split_data(
                df_mitigated,
                trial_seed,
                self.config.test_size / 100.0,
                self.config.valid_size / 100.0
            )
            del df_mitigated # Free memory
            PrintUtils.print_extra(f'Original Mitigated Split: Train={len(df_train_orig)}, Val={len(df_val)}, Test={len(df_test)}')
            PrintUtils.end_stage()

            # --- Downsample Training Data (if applicable) ---
            if sampling_rate < 1.0:
                df_train = df_train_orig.sample(frac=sampling_rate, random_state=trial_seed).reset_index(drop=True)
                PrintUtils.print_extra(f'Downsampled training set to {len(df_train)} samples ({sampling_rate*100:.1f}%)')
            else:
                df_train = df_train_orig.copy()
            del df_train_orig # Free memory


            # --- Apply Feature Mode ---
            # This step selects *which* columns (or combination) the Loader will use
            # by potentially modifying the columns *before* they go into the Loader.
            # Note: The Loader itself expects 'time_diffs' and 'data_lengths'
            #       as input for normalization calculation, even if only one is used later.
            #       The feature selection happens *conceptually* here and is enforced
            #       by how the model uses the data from the Loader.
            #       We'll pass the feature_mode to the model creation step.
            PrintUtils.print_extra(f"Preparing data for feature mode: {feature_mode.value}")


            # --- Prepare Data Loaders & Normalization ---
            PrintUtils.start_stage('Preparing data (Loaders/Normalization)')
            # Create datasets (Loader expects original column names for norm calc)
            train_dataset = Loader(df_train.copy()) # Pass copy to avoid modifying df_train
            val_dataset = Loader(df_val.copy())
            test_dataset = Loader(df_test.copy())

            # Get normalization params from the *sampled* training data
            norms = train_dataset.get_normalization() # Calculates max_len, means, stds etc.
            train_dataset.apply_normalization(norms)
            val_dataset.apply_normalization(norms)
            test_dataset.apply_normalization(norms)
            PrintUtils.end_stage()


            # --- Initialize Model ---
            # Pass feature_mode to potentially configure model architecture
            model = self.create_model(df_train, norms, feature_mode)
            if model is None:
                raise ValueError("Failed to create model.")


            # --- Train Model ---
            model_file_base = os.path.join(output_dir, 'model') # Base name for model files
            model_trainer = ModelTrainer(model, self.config, self.device)

            # Use unified fit method
            history = model_trainer.fit(
                train_data=train_dataset,
                val_data=val_dataset, # Use validation set for early stopping
                save_path=f"{model_file_base}.pth" # Path for final model state
            )


            # --- Evaluate Model ---
            PrintUtils.start_stage(f'Evaluating and visualizing for {run_desc}')

            # Get predictions on the test set using the *best* model state
            # The ModelTrainer should load the best state after fit if early stopping occurred
            test_scores, test_labels, _ = model_trainer.predict(test_dataset)
            if test_scores is None or test_labels is None:
                 raise ValueError("Prediction failed on the test set.")

            test_preds = (test_scores > 0.5).astype(int)

            # --- Create Visualization Plots ---
            # Only plot training curves if history is available (e.g., for PyTorch models)
            if isinstance(model, nn.Module) and history and 'train_losses' in history and history['train_losses']:
                try:
                    plot_training_curves(
                        history['train_losses'], history['val_losses'],
                        history['train_accs'], history['val_accs'],
                        history['best_epoch'], os.path.join(output_dir, f'training_curves.png')
                    )
                    create_model_dashboard(
                        test_scores, test_labels, history['train_losses'], history['val_losses'],
                        history['best_epoch'], os.path.join(output_dir, f'dashboard.png')
                    )
                except Exception as plot_err:
                    PrintUtils.print_warning(f"Could not generate training plots: {plot_err}")

            # Always generate evaluation plots
            try:
                plot_roc_curve(test_labels, test_scores, os.path.join(output_dir, f'roc_curve.png'))
                plot_precision_recall_curve(test_labels, test_scores, os.path.join(output_dir, f'pr_curve.png'))
                conf_matrix = plot_confusion_matrix(test_labels, test_preds, os.path.join(output_dir, f'confusion_matrix.png'))
                plot_score_distribution(test_scores, test_labels, os.path.join(output_dir, f'score_distribution.png'))
            except Exception as plot_err:
                 PrintUtils.print_warning(f"Could not generate evaluation plots: {plot_err}")
                 conf_matrix = confusion_matrix(test_labels, test_preds) # Calculate anyway if possible


            # --- Calculate and Store Metrics ---
            metrics = calculate_metrics(test_labels, test_scores, test_preds, conf_matrix, df_test)
            metrics['CommonName'] = common_name
            metrics['ChatBot'] = chatbot
            metrics['Mitigation Set'] = mitigation_set_name
            metrics['Features'] = feature_mode.value
            metrics['Trial'] = trial
            metrics['Sampling Rate'] = sampling_rate
            metrics['Best Epoch/Iter'] = history.get('best_epoch', getattr(model, 'best_iteration_', 0)) # Use model attr for LGBM
            metrics['Model Class'] = self.config.model_class
            metrics['Model Params'] = str(self.config.model_params)
            metrics['Train Params'] = str(self.config.training_params)
            metrics['Train Size'] = len(df_train)
            metrics['Val Size'] = len(df_val)
            metrics['Test Size'] = len(df_test)
            metrics['Data Path'] = data_path
            metrics['Median Time Before Mitigations'] = median_time
            metrics['Median Size Before Mitigations'] = median_size
            metrics['Median Time Delta Before Mitigations'] = median_time_delta
            metrics['Median Time After Mitigations'] = median_time_after_mitigations
            metrics['Median Size After Mitigations'] = median_size_after_mitigations
            metrics['Median Time Delta After Mitigations'] = median_time_delta_after_mitigations

            # Append to the main results list
            self.results.append(metrics)
            self._save_results_incrementally() # Save after each run

            # Save confusion matrix details
            self.save_confusion_metrics(test_labels, test_preds, conf_matrix, os.path.join(output_dir, f'confusion_matrix_metrics.txt'))

            # Save test predictions
            try:
                df_test_results = df_test.copy()
                df_test_results['prediction'] = test_preds
                df_test_results['score'] = test_scores
                # Select relevant columns to save
                cols_to_save = ['prompt', 'target', 'prediction', 'score']
                df_test_results[cols_to_save].to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
            except Exception as save_err:
                 PrintUtils.print_warning(f"Could not save test predictions: {save_err}")


            PrintUtils.print_extra(f'Completed: {run_desc}')
            PrintUtils.print_extra(f'Metrics - AUPRC: {metrics.get("AUPRC", np.nan):.4f}, F1: {metrics.get("F1 Score", np.nan):.4f}, Acc: {metrics.get("Accuracy", np.nan):.4f}')
            PrintUtils.end_stage() # End stage for this specific run processing

        except Exception as e:
            PrintUtils.end_stage(fail_message=f"Failed processing {run_desc}: {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            # Optionally add a failed run entry to results?
            # self.results.append({'ChatBot': chatbot, ..., 'Status': 'Failed', 'Error': str(e)})


    def _apply_feature_mode(self, df, feature_mode: FeatureMode):
        """
        DEPRECATED/REMOVED: This logic is now handled implicitly by how the
        model uses the data provided by the Loader and the `feature_mode`
        parameter during model initialization. The Loader always provides
        both normalized features if available.
        """
        # This function previously modified the DataFrame columns directly.
        # This is no longer the preferred way as the model itself should
        # select which features to use based on the `feature_mode` parameter.
        PrintUtils.print_extra(f"Feature mode '{feature_mode.value}' will be handled by the model architecture.")
        return df

    def _save_results_incrementally(self):
        """
        Save current results to CSV incrementally. This prevents data loss
        if the benchmark is interrupted.
        """
        if not self.results:
            return # Nothing to save

        try:
            results_df = pd.DataFrame(self.results)
            # Define preferred column order
            common_cols = [
                'CommonName', 'ChatBot', 'Mitigation Set', 'Features', 'Trial',
                'Sampling Rate', 'Data Path', 'Best Epoch/Iter', 'Model Class',
                'Train Size', 'Val Size', 'Test Size'
            ]
            metric_cols = [
                 'AUPRC', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity',
                 # Add other metrics here if calculated
            ]
            # Combine, ensuring existing columns are present
            ordered_cols = [col for col in common_cols + metric_cols if col in results_df.columns]
            # Add any remaining columns
            ordered_cols.extend([col for col in results_df.columns if col not in ordered_cols])

            results_df = results_df[ordered_cols]
            results_df.to_csv(self.results_file, index=False, float_format='%.4f')
        except Exception as e:
            PrintUtils.print_warning(f"Failed to save results incrementally: {e}")

    def create_model(self, df_train, normalization_params, feature_mode: FeatureMode):
        """
        Create model instance based on configuration and feature mode.

        Args:
            df_train: Training data (needed for BERT boundary calculation).
            normalization_params: Normalization parameters from Loader.
            feature_mode: The feature mode to use (Both, Data Size Only, Time Only).

        Returns:
            Initialized model instance, or None if creation fails.
        """
        PrintUtils.start_stage(f"Creating model: {self.config.model_class} for feature mode: {feature_mode.value}")

        model_instance = None
        model_class_name = self.config.model_class
        model_params = self.config.model_params.copy() # Use a copy

        try:
            # --- Adjust parameters based on feature mode if necessary ---
            # Example: Input dimensions for some models might change
            # For BERT, we might control interleaving or feature usage here
            if feature_mode == FeatureMode.DATA_SIZE_ONLY:
                PrintUtils.print_extra("Model should focus on data size features.")
                # Pass hint to model if it supports it, e.g., BERT might disable time tokens
                if model_class_name == 'BERTTimeSeriesClassifier':
                    model_params['single_feature'] = True
                    model_params['interleave'] = False # Ensure no interleaving
                # For LSTM/CNN, the Loader still provides 2 channels, but the model
                # might internally ignore one based on feature_mode if designed to.
                # Or, we rely on the _apply_feature_mode copy logic (less ideal).
                # Current BaseClassifier models expect 2 input channels from Loader.

            elif feature_mode == FeatureMode.TIME_ONLY:
                PrintUtils.print_extra("Model should focus on time difference features.")
                if model_class_name == 'BERTTimeSeriesClassifier':
                    model_params['single_feature'] = True
                    model_params['interleave'] = False
            else: # FeatureMode.BOTH
                 PrintUtils.print_extra("Model should use both time and size features.")
                 if model_class_name == 'BERTTimeSeriesClassifier':
                    model_params['single_feature'] = False
                    # 'interleave' could be True or False based on config


            # --- Instantiate the specific model class ---
            if model_class_name == 'LightGBMClassifier':
                # LightGBM handles features internally based on the prepared dataframe
                model_instance = LightGBMClassifier(
                    norm=normalization_params,
                    **model_params,
                )
                # Feature mode is conceptually handled by how data is prepared for LGBM
                # (though current Loader setup always gives both features to its internal prep)

            elif model_class_name == 'AttentionBiLSTMClassifier':
                model_instance = AttentionBiLSTMClassifier(
                    norm=normalization_params,
                    **model_params,
                )
                # Note: Assumes AttentionBiLSTMClassifier uses both input channels from Loader

            elif model_class_name == 'BERTTimeSeriesClassifier':
                # Calculate boundaries needed for BERT initialization
                # Ensure df_train has the necessary columns (might need reload or pass df_mitigated)
                time_boundaries_norm, len_boundaries_norm = BERTTimeSeriesClassifier.calculate_boundaries(
                    df_train, # Use the processed (mitigated, sampled) train set
                    num_buckets=model_params.get('num_buckets', 50),
                    norm=normalization_params
                )

                # Set default interleave based on feature mode if not specified
                if 'interleave' not in model_params:
                     model_params['interleave'] = (feature_mode == FeatureMode.BOTH)

                model_instance = BERTTimeSeriesClassifier(
                    normalization_params,
                    time_boundaries_norm,
                    len_boundaries_norm,
                    **model_params # Pass potentially modified params
                )

            # Add other model types here using elif
            # elif model_class_name == 'CNNClassifier': ...
            # elif model_class_name == 'LSTMTransformerClassifier': ...
            # elif model_class_name == 'CombinedLSTMBERTClassifier': ...

            else:
                raise ValueError(f"Unsupported model class: {model_class_name}")

            PrintUtils.print_extra(f"Model instance *{model_class_name}* created successfully.")
            PrintUtils.end_stage()
            return model_instance

        except Exception as e:
            PrintUtils.print_error(f"Error creating model {model_class_name}: {e}")
            PrintUtils.end_stage(fail_message=f"Model creation failed")
            import traceback
            traceback.print_exc()
            return None # Return None on failure


    def save_confusion_metrics(self, test_labels, test_preds, conf_matrix, output_file):
        """
        Save detailed confusion matrix metrics to a text file. Corrected calculation.

        Args:
            test_labels: True labels (numpy array).
            test_preds: Predicted labels (numpy array).
            conf_matrix: Confusion matrix (numpy array, [[TN, FP], [FN, TP]]).
            output_file: File to save the metrics to.
        """
        try:
            if conf_matrix.shape == (2, 2):
                tn, fp, fn, tp = conf_matrix.ravel()
            else: # Handle case where conf_matrix might be flattened or incorrect shape
                tn, fp, fn, tp = 0, 0, 0, 0 # Default to zero if matrix is invalid
                PrintUtils.print_warning("Invalid confusion matrix shape received. Metrics may be incorrect.")

            # Calculations with safe division
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            f1 = 2 * (precision_val * sensitivity) / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0.0

            with open(output_file, 'w') as f:
                f.write(f'Confusion Matrix:\n')
                f.write(f'                  Predicted Negative  Predicted Positive\n')
                f.write(f'Actual Negative      {tn:<18} {fp}\n')
                f.write(f'Actual Positive      {fn:<18} {tp}\n\n')

                f.write(f'Confusion Matrix Metrics:\n')
                f.write(f'Sensitivity/Recall: {sensitivity:.4f}\n')
                f.write(f'Specificity:        {specificity:.4f}\n')
                f.write(f'Precision:          {precision_val:.4f}\n')
                f.write(f'Negative Pred Value:{npv:.4f}\n')
                f.write(f'Accuracy:           {accuracy:.4f}\n')
                f.write(f'F1 Score:           {f1:.4f}\n')
        except Exception as e:
             PrintUtils.print_warning(f"Could not save confusion metrics to {output_file}: {e}")


    def reprocess_chatbot_statistics(self, chatbot, common_name, output_dir, prompts, feature_mode, trial):
        """
        Placeholder for reprocessing statistics without retraining.
        """
        PrintUtils.print_warning("Reprocessing stats without retraining is not implemented.")
        # Future implementation could load saved predictions/labels and recalculate/replot metrics.


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = ThrowingArgparse(description="Whisper Leak Benchmark Tool")
    parser.add_argument(
        '-c', '--config',
        help='Path to YAML configuration file (default: benchmark.yaml)',
        default='benchmark.yaml'
    )
    parser.add_argument(
        '-r', '--reprocess',
        help='Reprocess statistics without retraining models (not fully implemented)',
        action='store_true'
    )

    return parser.parse_args()


def main():
    """
    Main function to run the benchmark script
    """
    PrintUtils.print_logo() # Assuming PrintUtils has a logo function

    is_user_cancelled = False
    last_error = None

    try:
        args = parse_arguments()

        if not os.path.exists(args.config):
             raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
        PrintUtils.start_stage("Ensuring data directory exists")
        training_data_dir = Path(__file__).parent.parent / "data"
        training_data_dir.mkdir(parents=True, exist_ok=True)
        PrintUtils.end_stage()


        benchmark = BenchmarkRunner(args.config, reprocess_only=args.reprocess)
        benchmark.run_benchmark()

    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='Cancelled', throw_on_fail=False)
        PrintUtils.print_extra('\nOperation *cancelled* by user.')
        is_user_cancelled = True

    except FileNotFoundError as fnf_error:
         if PrintUtils.is_in_stage():
              PrintUtils.end_stage(fail_message=str(fnf_error), throw_on_fail=False)
         PrintUtils.print_error(f'Error: {fnf_error}')
         last_error = fnf_error

    except Exception as ex:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=str(ex), throw_on_fail=False) # End stage gracefully
        PrintUtils.print_error(f'An unexpected error occurred: {ex}')
        import traceback
        traceback.print_exc() # Print detailed traceback
        last_error = ex

    finally:
        # Print final status message
        if last_error is not None:
            PrintUtils.print_error(f'\nBenchmark finished with ERRORS.')
        elif is_user_cancelled:
            PrintUtils.print_extra(f'\nBenchmark *cancelled* by user.')
        else:
            PrintUtils.print_extra(f'\nBenchmark completed successfully.')


if __name__ == '__main__':
    main()
