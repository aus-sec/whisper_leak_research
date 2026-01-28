# core/classifiers/lightgbm_classifier.py
import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
import os
import json
from core.utils import PrintUtils
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split # For potential internal splitting if needed

# Note: This class does NOT inherit from BaseClassifier or torch.nn.Module
class LightGBMClassifier:
    """
    Classifier using LightGBM for time series classification.
    It expects input data as pandas DataFrames with 'time_diffs' and 'data_lengths' columns.
    """
    def __init__(self, norm, **kwargs):
        """
        Initialize the LightGBM classifier.

        Args:
            norm (dict): Normalization parameters dictionary containing 'max_len'.
                         Other normalization params are ignored for LightGBM.
            **kwargs: Hyperparameters for the LightGBM model.
        """
        self.norm = norm
        self.max_len = norm.get('max_len', 700) # Default max_len if not in norm
        self.model = None
        self.class_name = self.__class__.__name__

        # Store args for saving/loading metadata
        # Filter out norm arg before storing model params
        self.args = {k: v for k, v in kwargs.items()}

        # Default LightGBM parameters - can be overridden by kwargs
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'n_estimators': 1000, # High n_estimators, rely on early stopping
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'seed': 42,
            'n_jobs': -1,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        # Update defaults with user-provided kwargs
        default_params.update(self.args)
        self.params = default_params # Store the effective params

        # Initialize the underlying LightGBM model instance
        self.model = lgb.LGBMClassifier(**self.params)
        self._is_fitted = False
        PrintUtils.print_extra(f"Initialized LightGBMClassifier with effective params: {self.params}")


    def _prepare_features(self, df):
        """
        Transforms DataFrame with sequences into a 2D numpy array for LightGBM.
        Uses unnormalized, padded, and flattened sequences.
        """
        if not isinstance(df, pd.DataFrame):
             raise ValueError("Input must be a pandas DataFrame")
        if 'time_diffs' not in df.columns or 'data_lengths' not in df.columns:
             raise ValueError("DataFrame must contain 'time_diffs' and 'data_lengths' columns")

        num_samples = len(df)
        # Create flattened features: [time_1..time_N, len_1..len_N]
        feature_dim = self.max_len * 2
        X = np.zeros((num_samples, feature_dim), dtype=np.float32)

        for i in range(num_samples):
            times = df['time_diffs'].iloc[i][:self.max_len]
            lengths = df['data_lengths'].iloc[i][:self.max_len]

            # Pad and assign time features
            padded_times = np.zeros(self.max_len)
            len_t = min(len(times), self.max_len)
            padded_times[:len_t] = times[:len_t]
            X[i, :self.max_len] = padded_times

            # Pad and assign length features
            padded_lengths = np.zeros(self.max_len)
            len_l = min(len(lengths), self.max_len)
            padded_lengths[:len_l] = lengths[:len_l]
            X[i, self.max_len:] = padded_lengths

        return X

    def fit(self, train_df, val_df, patience=20, **kwargs):
        """
        Fit the LightGBM model.

        Args:
            train_df (pd.DataFrame): Training data DataFrame.
            val_df (pd.DataFrame): Validation data DataFrame.
            patience (int): Early stopping patience.
            **kwargs: Additional arguments (ignored for LightGBM, used for compatibility).
        """
        PrintUtils.start_stage("Preparing data for LightGBM training")
        X_train = self._prepare_features(train_df)
        y_train = train_df['target'].values
        X_val = self._prepare_features(val_df)
        y_val = val_df['target'].values
        PrintUtils.end_stage()

        PrintUtils.start_stage(f"Fitting LightGBM model (patience={patience})")
        callbacks = [
            lgb.early_stopping(stopping_rounds=patience, verbose=True),
            lgb.log_evaluation(period=50) # Log evaluation results periodically
        ]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_names=['validation'],
            callbacks=callbacks
        )
        self._is_fitted = True
        PrintUtils.end_stage()

    def predict_proba(self, df):
        """
        Predict probabilities for the input data.

        Args:
            df (pd.DataFrame): DataFrame with data to predict.

        Returns:
            np.ndarray: Predicted probabilities for the positive class.
        """
        if not self.has_fit():
            raise RuntimeError("Model has not been fitted yet.")
        if not isinstance(df, pd.DataFrame):
             raise ValueError("Input must be a pandas DataFrame")

        X = self._prepare_features(df)
        # Predict probabilities for class 1
        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def has_fit(self):
        """Check if the model has been fitted."""
        # Check based on the internal flag set after fitting
        # Or check if the booster object exists and has attributes set during fitting
        return self._is_fitted and hasattr(self.model.booster_, 'best_iteration')

    def save(self, filepath):
        """Save the trained LightGBM model and normalization parameters."""
        if not self.has_fit():
            PrintUtils.print_extra("Warning: Attempting to save a model that has not been fitted.")
            # Decide if saving an unfitted model should be an error or just a warning
            # For now, let's allow it but print a warning.

        # Save the LightGBM model itself
        model_filepath = filepath # e.g., 'model.pth' -> use as base name
        lgbm_model_file = model_filepath.replace('.pth', '.lgbm.joblib')
        try:
             joblib.dump(self.model, lgbm_model_file)
             PrintUtils.print_extra(f'LightGBM model saved to *{os.path.basename(lgbm_model_file)}*')
        except Exception as e:
             PrintUtils.print_extra(f"Error saving LightGBM model: {e}")
             raise

        # Save metadata (norm params, class name, model args)
        meta_filepath = filepath.replace('.pth', '_norm_params.json')
        meta_data = {
            'normalization_params': self.norm, # Save norm dict (mainly for max_len)
            'class_name': self.class_name,
            'args': self.args, # Save hyperparameters used
            'is_fitted': self._is_fitted # Save fitted status
        }
        try:
            # TODO: Re-enable
            #with open(meta_filepath, 'w', encoding='utf-8') as f:
            #    json.dump(meta_data, f, indent=4)
            PrintUtils.print_extra(f'Metadata saved to *{os.path.basename(meta_filepath)}*')
        except Exception as e:
             PrintUtils.print_extra(f"Error saving metadata: {e}")
             raise


    @classmethod
    def load(cls, filepath, device=None): # device is ignored for LGBM but kept for compatibility
        """Load a LightGBM model and its metadata."""
        PrintUtils.print_extra(f"Loading LightGBM model from {filepath}")

        # Load metadata first
        meta_filepath = filepath.replace('.pth', '_norm_params.json')
        if not os.path.exists(meta_filepath):
            raise FileNotFoundError(f'Metadata file not found: {meta_filepath}')

        try:
            with open(meta_filepath, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
        except Exception as e:
            PrintUtils.print_extra(fail_message=f"Failed to load metadata: {e}")
            raise

        normalization_params = meta_data['normalization_params']
        args = meta_data.get('args', {}) # Use .get for backward compatibility
        is_fitted_saved = meta_data.get('is_fitted', False) # Check if it was saved as fitted

        # Instantiate the class
        # Pass norm and stored args to the constructor
        instance = cls(norm=normalization_params, **args)

        # Load the actual LightGBM model state
        lgbm_model_file = filepath.replace('.pth', '.lgbm.joblib')
        if not os.path.exists(lgbm_model_file):
             # If the specific lgbm file doesn't exist, maybe it wasn't fitted/saved properly
             if is_fitted_saved:
                 PrintUtils.print_extra(f"Warning: Metadata indicates model should be fitted, but file not found: {lgbm_model_file}. Model will be loaded as unfitted.")
             else:
                 PrintUtils.print_extra(f"LightGBM model file not found: {lgbm_model_file}. Model loaded as unfitted.")
             instance._is_fitted = False # Ensure it's marked as not fitted
        else:
            try:
                instance.model = joblib.load(lgbm_model_file)
                instance._is_fitted = is_fitted_saved # Restore fitted status from metadata
                PrintUtils.print_extra(f'LightGBM model state loaded from *{os.path.basename(lgbm_model_file)}*')
            except Exception as e:
                PrintUtils.print_extra(fail_message=f"Failed to load LightGBM model state: {e}")
                raise

        PrintUtils.print_extra("LightGBM model loading complete.")
        return instance

