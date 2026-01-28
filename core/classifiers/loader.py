import bisect
import pandas as pd
import torch
from core.utils import PrintUtils

import ast
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import random

class Loader(Dataset):
    """
        Dataset class for preprocessed time series data.
    """

    def __init__(self, df):
        """
            Creates an instance.
        """

        # Save members
        self.df = df.reset_index(drop=True)
        self.df = self.df[ ['prompt', 'time_diffs', 'data_lengths', 'target'] ]

        # Validate columns time_diffs, data_lengths, and target
        if 'time_diffs' not in df.columns or 'data_lengths' not in df.columns or 'target' not in df.columns:
            raise ValueError('Dataframe must contain columns: time_diffs, data_lengths, and target.')

        self.normalized = False

    def __len__(self):
        """
            Length override.
        """

        # Return the dataframe length
        return len(self.df)
    
    def get_normalization(self):
        """
        Normalizes time_diffs and data_lengths uses Z-score normalization (standardization).

        Returns: Dict, keys time_mean, time_std, size_mean, size_std, max_len
        """
        # Preprocess datasets
        self.df.loc[:, 'time_diffs'] = self.df['time_diffs'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.df.loc[:, 'data_lengths'] = self.df['data_lengths'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.max_len = int(np.percentile(self.df['data_lengths'].apply(len), 95))
        
        # Flatten the lists to calculate global statistics
        all_time_vals = [val for row in self.df['time_diffs'] for val in row]
        all_size_vals = [val for row in self.df['data_lengths'] for val in row]
        
        # Calculate mean and standard deviation for time_diffs
        time_mean = np.mean(all_time_vals)
        time_std = np.std(all_time_vals)
        if time_std <= 0:
            time_std = 1.0  # Avoid division by zero
        
        # Calculate mean and standard deviation for data_lengths
        size_mean = np.mean(all_size_vals)
        size_std = np.std(all_size_vals)
        if size_std <= 0:
            size_std = 1.0  # Avoid division by zero

        # Log the normalization parameters
        PrintUtils.print_extra(f'Global time normalization: mean=*{time_mean:.4f}*, std=*{time_std:.4f}*')
        PrintUtils.print_extra(f'Global size normalization: mean=*{size_mean:.4f}*, std=*{size_std:.4f}*')
        
        # Compute rank normalization for time_diffs and data_lengths
        time_value_map, time_unique_set = self._get_normalize_ranks(all_time_vals)
        size_value_map, size_unique_set = self._get_normalize_ranks(all_size_vals)

        norm = {}
        norm['time_value_map'] = time_value_map
        norm['time_unique_set'] = time_unique_set
        norm['size_value_map'] = size_value_map
        norm['size_unique_set'] = size_unique_set
        norm['max_len'] = self.max_len
        norm['time_mean'] = time_mean
        norm['time_std'] = time_std
        norm['size_mean'] = size_mean
        norm['size_std'] = size_std

        return norm
    
    def apply_normalization(self, norm):
        """
        Normalize Z-score normalization and rank normalization for time_diffs and data_lengths.
        """

        self.max_len = norm['max_len']
        
        # Create deep copy to avoid modifying the original dataframe
        df_normalized = self.df.copy()
        
        # Apply z-score normalization
        z_time_diffs = []
        z_data_lengths = []
        
        for idx, row in self.df.iterrows():
            time_vals, size_vals = row['time_diffs'], row['data_lengths']
            
            # Normalize time_diffs
            norm_time = [(val - norm['time_mean']) / norm['time_std'] for val in time_vals]
            z_time_diffs.append(norm_time)
            
            # Normalize data_lengths
            norm_size = [(val - norm['size_mean']) / norm['size_std'] for val in size_vals]
            z_data_lengths.append(norm_size)

        # Apply rank normalization using the training value maps
        rank_time_diffs = df_normalized['time_diffs'].apply(
            lambda x: self._apply_normalize_ranks(x, norm['time_value_map'], norm['time_unique_set'])
        )
        rank_data_lengths = df_normalized['data_lengths'].apply(
            lambda x: self._apply_normalize_ranks(x, norm['size_value_map'], norm['size_unique_set'])
        )
        
        # Add normalized features to the dataframe
        df_normalized['time_diffs_z_norm'] = z_time_diffs
        df_normalized['data_lengths_z_norm'] = z_data_lengths
        df_normalized['time_diffs_rank_norm'] = rank_time_diffs
        df_normalized['data_lengths_rank_norm'] = rank_data_lengths
        self.df = df_normalized
        self.normalized = True
        
        PrintUtils.end_stage()
        return
    
    def _get_normalize_ranks(self, training_flat_values):
        """
        Calculates the normalization rank map and sorted unique values from
        training data, aiming for a uniform distribution in the range [-1, 1].

        Args:
            training_flat_values: A single list containing all numeric training
                                 values concatenated from all original rows.
                                 NaNs are permissible.

        Returns:
            A tuple containing:
            1. value_to_rank_map (Dict): Maps each unique non-NaN training value
                                         to its calculated scaled rank (-1 to 1).
                                         Duplicate training values get the same rank.
            2. sorted_unique_training_values (List): A sorted list of unique,
                                                     non-NaN values from the training set.
            Returns ({}, []) if input is empty or contains only NaNs.
        """
        if not training_flat_values:
            print("Warning: Input 'training_flat_values' is empty. Returning empty results.")
            return {}, []

        # Convert to pandas Series for efficient ranking
        values_series = pd.Series(training_flat_values, dtype=float) # Ensure float type

        # Filter out NaNs for creating the sorted unique list and the value->rank map
        non_nan_series = values_series.dropna()
        if non_nan_series.empty:
            print("Warning: Input 'training_flat_values' contains only NaNs. Returning empty results.")
            return {}, []

        # Calculate ranks as percentiles for the original series (including NaNs)
        percentile_ranks = values_series.rank(method='average', pct=True, na_option='keep')

        # Scale ranks from [~0, 1] to [-1, 1]
        scaled_ranks = (percentile_ranks * 2) - 1

        # Create the mapping: {original_value: scaled_rank}
        # Use the non_nan_series index to correctly map original values (handling duplicates)
        value_to_rank_map = {}
        for idx, original_value in values_series.items():
             # Only map non-NaN values; ensure a value isn't overwritten by a later NaN rank
             if pd.notna(original_value) and pd.notna(scaled_ranks[idx]):
                 # If a value appears multiple times, rank() with 'average' gives them the same rank
                 # so overwriting in the dict is fine and achieves the desired result.
                 value_to_rank_map[original_value] = scaled_ranks[idx]


        # Get sorted unique non-NaN values
        sorted_unique_training_values = sorted(non_nan_series.unique())

        return value_to_rank_map, sorted_unique_training_values


    def _apply_normalize_ranks(self, inference_values, training_value_to_rank_map, sorted_unique_training_values):
        """
        Normalizes a list of inference values using ranks derived from training data.
        If an inference value matches a training value, its rank is used directly.
        Otherwise, the rank of the nearest training value is used.

        Args:
            inference_values: A list of numeric values from a single inference row.
            training_value_to_rank_map: The dictionary mapping training value to
                                        scaled rank, from _get_normalize_ranks.
            sorted_unique_training_values: The sorted list of unique non-NaN
                                           training values, from _get_normalize_ranks.

        Returns:
            A list containing the normalized values (scaled ranks between -1 and 1)
            corresponding to the inference values. Returns NaN for NaN inputs.
            Returns NaN if training data was empty/all NaNs.
        """
        normalized_output = []

        # Handle case where training data yielded no ranks/values
        if not sorted_unique_training_values or not training_value_to_rank_map:
            print("Warning: Training rank map or sorted values are empty. Returning NaNs.")
            return [np.nan] * len(inference_values)

        n_training = len(sorted_unique_training_values)

        for value in inference_values:
            # Handle NaN input
            if pd.isna(value):
                normalized_output.append(np.nan)
                continue

            # Check for exact match in training values
            if value in training_value_to_rank_map:
                normalized_output.append(training_value_to_rank_map[value])
                continue

            # --- Find nearest training value if no exact match ---
            # Use binary search (bisect_left) to find insertion point
            insertion_point = bisect.bisect_left(sorted_unique_training_values, value)

            # Determine nearest neighbor(s)
            if insertion_point == 0:
                # Closer to the first element
                nearest_training_value = sorted_unique_training_values[0]
            elif insertion_point == n_training:
                # Closer to the last element
                nearest_training_value = sorted_unique_training_values[-1]
            else:
                # Between two elements, find the closer one
                val_before = sorted_unique_training_values[insertion_point - 1]
                val_after = sorted_unique_training_values[insertion_point]
                if abs(value - val_before) <= abs(value - val_after):
                    nearest_training_value = val_before
                else:
                    nearest_training_value = val_after

            # Get the rank of the nearest training value
            # Use .get() for safety, though nearest_training_value should always be in the map
            nearest_rank = training_value_to_rank_map.get(nearest_training_value, np.nan)
            normalized_output.append(nearest_rank)

        return normalized_output

    def __getitem__(self, idx):
        """
            Accessor override.
        """
        if not self.normalized:
            raise ValueError('Data must be normalized before accessing items.')

        # Get the time and data lengths
        row = self.df.iloc[idx]
        time_z = row['time_diffs_z_norm']
        size_z = row['data_lengths_z_norm']
        time_rank = row['time_diffs_rank_norm']
        size_rank = row['data_lengths_rank_norm']

        # Pad sequences to max_len
        time_z_padded = np.zeros(self.max_len)
        size_z_padded = np.zeros(self.max_len)
        time_rank_padded = np.zeros(self.max_len)
        size_rank_padded = np.zeros(self.max_len)
        time_z_padded[:len(time_z)] = time_z[:self.max_len]
        size_z_padded[:len(time_z)] = size_z[:self.max_len]
        time_rank_padded[:len(time_rank)] = time_rank[:self.max_len]
        size_rank_padded[:len(size_rank)] = size_rank[:self.max_len]

        # Return the tensor
        #sample = np.stack([time_z_padded, size_z_padded, time_rank_padded, size_rank_padded], axis=0)
        sample = np.stack([time_z_padded, size_z_padded], axis=0)
        target = row['target']
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
