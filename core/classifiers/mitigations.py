# core/mitigations.py

import abc
from collections import deque
import math
import random
import numpy as np
import pandas as pd
import ast
from core.utils import PrintUtils # Assuming PrintUtils handles logging/printing

class BaseMitigation(abc.ABC):
    """
    Abstract base class for all mitigation strategies.

    Mitigations transform the input DataFrame (containing 'time_diffs'
    and 'data_lengths' columns) before normalization and model training.
    """

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the mitigation logic to the DataFrame.

        Args:
            df: The input DataFrame with 'time_diffs' and 'data_lengths' columns.
                These columns should contain lists of numbers.

        Returns:
            A new DataFrame with the mitigation applied.
        """
        pass

    def _ensure_list_format(self, series: pd.Series) -> pd.Series:
        """
        Ensures that the elements in the series are lists, attempting conversion
        from string representation if necessary.

        Args:
            series: A pandas Series, potentially containing string representations
                    of lists.

        Returns:
            A pandas Series where elements are lists.
        """
        # Check if the first element is already a list and non-empty
        first_valid = series.dropna().iloc[0] if not series.dropna().empty else None
        if first_valid is not None and isinstance(first_valid, list):
            return series

        # Attempt conversion using ast.literal_eval
        PrintUtils.print_extra(f"Attempting list conversion for column '{series.name}'")
        try:
            return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))
        except (ValueError, SyntaxError, TypeError) as e:
            PrintUtils.print_extra(f"Could not parse column '{series.name}' elements as lists: {e}. Mitigation might fail or be skipped.")
            # Return original but ensure elements are lists (even if empty on failure)
            return series.apply(lambda x: x if isinstance(x, list) else [])


class PacketInjectionMitigation(BaseMitigation):
    """
    Mitigation that injects noise packets into the sequence based on timing intervals.
    
    Injections occur at intervals based on mean_diff/injection_multiplier_mean with
    normal distribution variance. The size of injected packets uses an adaptive
    approach that considers whether packet sizes are monotonically increasing.
    
    For monotonic sequences: last_size + normal(mean_size_increase, std_increase)
    For non-monotonic sequences: normal(mean_size, std_size)
    """
    def __init__(self,
                 injections_per_mean: float = 2.0,
                 injection_stddev_multiplier: float = 2.0,
                 monotonic_threshold: float = 0.9,
                 size_std_multiplier: float = 1.0,
                 max_injections_per_packet: int = 10):
        """
        Initializes the packet injection mitigation.

        Args:
            injections_per_mean: Mean for injection timing (mean_diff/this_value). Defaults to 2.0.
            injection_stddev_multiplier: Standard deviation for injection timing. Defaults to 2.0.
            monotonic_threshold: Threshold for considering sizes monotonically increasing (0.0-1.0). Defaults to 0.9.
            size_std_multiplier: Multiplier for standard deviation in normal distribution sizing. Defaults to 1.0.
            max_injections_per_packet: Maximum number of injections allowed per original packet. Defaults to 10.
        """
        if injections_per_mean <= 0:
            raise ValueError("Injection multiplier mean must be positive.")
        if injection_stddev_multiplier < 0:
            raise ValueError("Injection multiplier stdv cannot be negative.")
        if not (0.0 <= monotonic_threshold <= 1.0):
            raise ValueError("Monotonic threshold must be between 0.0 and 1.0.")
        if size_std_multiplier < 0:
            raise ValueError("Size std multiplier cannot be negative.")
        if max_injections_per_packet < 1:
            raise ValueError("Max injections per packet must be at least 1.")

        self.injections_per_mean = injections_per_mean
        self.injection_stddev_multiplier = injection_stddev_multiplier
        self.monotonic_threshold = monotonic_threshold
        self.size_std_multiplier = size_std_multiplier
        self.max_injections_per_packet = max_injections_per_packet

        # Initialize statistics that will be calculated during transform
        self.mean_size = None
        self.std_size = None
        self.mean_size_increase = None
        self.std_size_increase = None
        self.is_monotonic = None

        PrintUtils.print_extra(
            f"Initialized PacketInjectionMitigation with mult_mean={self.injections_per_mean:.2f}, "
            f"mult_stdv={self.injection_stddev_multiplier:.2f}, "
            f"monotonic_threshold={self.monotonic_threshold:.2f}, "
            f"max_injections_per_packet={self.max_injections_per_packet}"
        )

    def _calculate_time_stats(self, series: pd.Series) -> tuple[float, float]:
        """Calculates mean and stddev of all non-zero time values."""
        try:
            all_times = [
                float(time) for sublist in series if isinstance(sublist, list)
                for time in sublist if isinstance(time, (int, float)) and not math.isnan(time)
            ]
            non_zero_times = [time for time in all_times if time > 0.005]

            if not non_zero_times:
                return 0.0, 0.0
            return float(np.mean(non_zero_times)), float(np.std(non_zero_times))
        except Exception as e:
            PrintUtils.print_extra(f"Could not calculate time statistics: {e}")
            return 0.0, 0.0


    def _remove_zero_time_entries(self, times: list, sizes: list) -> tuple[list, list]:
        """
        Removes entries with zero time differences and combines their sizes
        with the previous entry. This should be applied before injection analysis.
        
        Args:
            times: List of inter-packet time differences.
            sizes: List of data lengths.
            
        Returns:
            Tuple of cleaned time and size lists.
        """
        if not times or len(times) != len(sizes):
            return times, sizes
            
        cleaned_times = []
        cleaned_sizes = []
        
        for i, (t, s) in enumerate(zip(times, sizes)):
            # Convert to numeric and validate
            try:
                time_val = float(t) if isinstance(t, (int, float)) and not math.isnan(t) else 0.0
                size_val = int(s) if isinstance(s, (int, float)) and not math.isnan(s) else 0
            except (ValueError, TypeError):
                time_val, size_val = 0.0, 0
                
            # If time is significant (> 0.005), keep as separate entry
            if time_val > 0.005:
                cleaned_times.append(time_val)
                cleaned_sizes.append(size_val)
            # If time is near zero and we have previous entries, combine with last entry
            elif cleaned_times:
                cleaned_sizes[-1] += size_val
                # Don't add to cleaned_times since this packet arrived simultaneously
            # If it's the first packet and has zero time, still keep it
            else:
                cleaned_times.append(max(0.0, time_val))
                cleaned_sizes.append(size_val)
                
        return cleaned_times, cleaned_sizes

    def _analyze_size_monotonicity(self, sizes: list) -> bool:
        """
        Analyzes whether the size sequence is mostly monotonically increasing.
        
        Args:
            sizes: List of packet sizes.
            
        Returns:
            True if the percentage of increasing consecutive pairs exceeds the threshold.
        """
        if len(sizes) < 2:
            return False
            
        increasing_count = 0
        total_pairs = 0
        
        for i in range(len(sizes) - 1):
            try:
                current_size = int(sizes[i]) if isinstance(sizes[i], (int, float)) and not math.isnan(sizes[i]) else 0
                next_size = int(sizes[i + 1]) if isinstance(sizes[i + 1], (int, float)) and not math.isnan(sizes[i + 1]) else 0
                
                if next_size > current_size:
                    increasing_count += 1
                total_pairs += 1
            except (ValueError, TypeError):
                continue
                
        if total_pairs == 0:
            return False
            
        monotonic_ratio = increasing_count / total_pairs
        return monotonic_ratio >= self.monotonic_threshold

    def _calculate_size_stats(self, series: pd.Series) -> tuple[float, float, float, float, bool]:
        """
        Calculates comprehensive size statistics for adaptive injection.
        
        Args:
            series: Pandas Series containing lists of packet sizes.
            
        Returns:
            Tuple of (mean_size, std_size, mean_increase, std_increase, is_monotonic)
        """
        try:
            all_sizes = []
            all_increases = []
            monotonic_sequences = 0
            total_sequences = 0
            
            for sublist in series:
                if not isinstance(sublist, list) or len(sublist) < 1:
                    continue
                    
                # Convert to numeric and filter valid sizes
                sizes = []
                for size in sublist:
                    try:
                        size_val = int(size) if isinstance(size, (int, float)) and not math.isnan(size) else 0
                        if size_val > 0:
                            sizes.append(size_val)
                    except (ValueError, TypeError):
                        continue
                
                if not sizes:
                    continue
                    
                all_sizes.extend(sizes)
                total_sequences += 1
                
                # Analyze monotonicity and calculate increases for this sequence
                if self._analyze_size_monotonicity(sizes):
                    monotonic_sequences += 1
                    
                # Calculate size increases for this sequence
                for i in range(len(sizes) - 1):
                    increase = sizes[i + 1] - sizes[i]
                    if increase > 0:  # Only consider positive increases
                        all_increases.append(increase)
            
            # Calculate overall statistics
            mean_size = float(np.mean(all_sizes)) if all_sizes else 0.0
            std_size = float(np.std(all_sizes)) if len(all_sizes) > 1 else 0.0
            mean_increase = float(np.mean(all_increases)) if all_increases else 0.0
            std_increase = float(np.std(all_increases)) if len(all_increases) > 1 else 0.0
            
            # Overall monotonicity based on sequence-level analysis
            is_monotonic = (monotonic_sequences / total_sequences >= self.monotonic_threshold) if total_sequences > 0 else False
            
            return mean_size, std_size, mean_increase, std_increase, is_monotonic
            
        except Exception as e:
            PrintUtils.print_extra(f"Could not calculate size statistics: {e}")
            return 0.0, 0.0, 0.0, 0.0, False

    def _get_injection_size(self, last_size: int = None) -> int:
        """
        Determines the size of the packet to be injected using adaptive normal distribution.
        
        Args:
            last_size: The size of the most recent packet.
            
        Returns:
            The size of the injected packet.
        """
        if self.is_monotonic and last_size is not None and self.mean_size_increase is not None:
            # For monotonic sequences: last_size + normal(mean_increase, std_increase)
            increase = np.random.normal(
                self.mean_size_increase, 
                self.std_size_increase * self.size_std_multiplier
            )
            injection_size = last_size + increase
        else:
            # For non-monotonic sequences: normal(mean_size, std_size)
            injection_size = np.random.normal(
                self.mean_size if self.mean_size is not None else 64,
                self.std_size * self.size_std_multiplier if self.std_size is not None else 32
            )
        
        # Ensure positive size with minimum value
        return max(1, int(round(injection_size)))


    def _get_next_injection_time(self) -> float:
        """
        Sample next injection interval from N(mean, stddev) using dataset stats.
        Uses percentage-based standard deviation to maintain consistent median timing.
        """
        base_mean = self.mean_time_diff / self.injections_per_mean
        # Use percentage of mean for standard deviation to avoid excessive variance
        base_std = base_mean * (self.injection_stddev_multiplier * 0.1)  # 0.1 = 10% per unit
        interval = np.random.normal(base_mean, base_std)
        return max(0.001, interval)  # prevent near-zero times that skew median


    def _apply_injection(self, original_times: list, original_sizes: list) -> tuple[list, list]:
        """
        Applies packet injection logic to time and size sequences.
        First applies RemoveZeroTimeEntries preprocessing, then performs injection.
        Limits the number of injections per original packet to max_injections_per_packet.
        """
        if not isinstance(original_times, list) or not isinstance(original_sizes, list):
            return original_times, original_sizes
        if len(original_times) != len(original_sizes) or not original_times:
            return original_times, original_sizes
        if self.mean_time_diff is None or self.mean_time_diff <= 0.00001:
            return original_times, original_sizes
        if self.mean_size is None or self.mean_size <= 0:
            return original_times, original_sizes

        # Step 1: Apply RemoveZeroTimeEntries preprocessing
        cleaned_times, cleaned_sizes = self._remove_zero_time_entries(original_times, original_sizes)
        
        if not cleaned_times:
            return [], []

        # Ensure numeric types
        times_numeric = [(float(t) if isinstance(t, (int, float)) and not math.isnan(t) else 0.0) for t in cleaned_times]
        sizes_numeric = [(int(s) if isinstance(s, (int, float)) and not math.isnan(s) else 0) for s in cleaned_sizes]

        new_times = []
        new_sizes = []
        
        cumulative_original_time = 0.0
        cumulative_emitted_time = 0.0
        next_injection_time = self._get_next_injection_time()
        
        for orig_time, orig_size in zip(times_numeric, sizes_numeric):
            cumulative_original_time += orig_time
            
            # Track injections for this packet
            injections_this_packet = 0
            
            # Check for injection opportunities before this packet
            while (next_injection_time <= (cumulative_original_time - cumulative_emitted_time) and 
                   injections_this_packet < self.max_injections_per_packet):
                # Get the last size for adaptive injection
                last_size = new_sizes[-1] if new_sizes else (sizes_numeric[0] if sizes_numeric else None)
                
                # Inject a packet
                new_times.append(next_injection_time)
                new_sizes.append(self._get_injection_size(last_size))
                cumulative_emitted_time += next_injection_time
                next_injection_time = self._get_next_injection_time()
                injections_this_packet += 1
            
            # Emit the original packet
            time_to_emit = cumulative_original_time - cumulative_emitted_time
            new_times.append(time_to_emit)
            new_sizes.append(orig_size)
            cumulative_emitted_time = cumulative_original_time

        return new_times, new_sizes

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies packet injection to the 'time_diffs' and 'data_lengths' columns.

        Args:
            df: Input DataFrame with 'time_diffs' and 'data_lengths' columns.

        Returns:
            A new DataFrame with the mitigation applied.
        """
        mitigation_name = self.__class__.__name__
        PrintUtils.start_stage(f"Applying {mitigation_name}")
        df_copy = df.copy()

        # Column checks
        if 'time_diffs' not in df_copy.columns or 'data_lengths' not in df_copy.columns:
            PrintUtils.print_extra(f"Required columns missing. Skipping {mitigation_name}.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure list format
        try:
            df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])
            df_copy['data_lengths'] = self._ensure_list_format(df_copy['data_lengths'])
        except Exception as e:
            PrintUtils.print_error(f"Error ensuring list format: {e}. Skipping mitigation.")
            PrintUtils.end_stage(fail_message="List format conversion failed.")
            return df

        # Calculate dataset-wide statistics
        self.mean_time_diff, self.stdv_time_diff = self._calculate_time_stats(df_copy['time_diffs'])
        PrintUtils.print_extra(f"Calculated time stats - mean: {self.mean_time_diff:.6f} s, stdv: {self.stdv_time_diff:.6f} s")

        # Calculate comprehensive size statistics for adaptive approach
        self.mean_size, self.std_size, self.mean_size_increase, self.std_size_increase, self.is_monotonic = self._calculate_size_stats(df_copy['data_lengths'])
        PrintUtils.print_extra(f"Calculated adaptive size stats - mean: {self.mean_size:.2f} bytes, std: {self.std_size:.2f} bytes")
        PrintUtils.print_extra(f"Size increase stats - mean: {self.mean_size_increase:.2f} bytes, std: {self.std_size_increase:.2f} bytes")
        PrintUtils.print_extra(f"Dataset is {'monotonic' if self.is_monotonic else 'non-monotonic'} (threshold: {self.monotonic_threshold:.1%})")

        # Check if we can apply mitigation
        can_apply = self.mean_time_diff is not None and self.mean_time_diff > 0.001
        can_apply &= (self.mean_size is not None and self.mean_size > 0)

        if not can_apply:
            PrintUtils.print_extra(f"Cannot apply {mitigation_name} due to invalid calculations.")
            PrintUtils.end_stage()
            return df_copy

        # Apply mitigation row-wise
        new_times_list = []
        new_sizes_list = []
        for index, row in df_copy.iterrows():
            try:
                new_t, new_s = self._apply_injection(row['time_diffs'], row['data_lengths'])
            except Exception as e:
                PrintUtils.print_extra(f"Error applying injection for row {index}: {e}")
                new_t, new_s = row['time_diffs'], row['data_lengths']

            new_times_list.append(new_t)
            new_sizes_list.append(new_s)

        df_copy['time_diffs'] = new_times_list
        df_copy['data_lengths'] = new_sizes_list

        PrintUtils.end_stage()
        return df_copy


class RemoveZeroTimeEntries(BaseMitigation):
    """
    Mitigation removes entries with zero time differences from the DataFrame. The
    corresponding data lengths are added to the previous entry's data length.
    """
    def __init__(self):
        """
        Initializes the random time delay mitigation.
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies random delays to the 'time_diffs' column.

        Args:
            df: The input DataFrame.
            median_time_delta: Pre-calculated median time delta from the full dataset.
            median_data_size: Pre-calculated median data size from the full dataset.

        Returns:
            DataFrame with random delays added to 'time_diffs'.
        """
        PrintUtils.start_stage(f"Applying RemoveZeroTimeEntries")
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns:
            PrintUtils.print_extra("Column 'time_diffs' not found. Skipping RemoveZeroTimeEntries.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure time_diffs are lists
        df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])

        for index, row in df_copy.iterrows():
            times = row['time_diffs']
            sizes = row['data_lengths'] if 'data_lengths' in df_copy.columns else None

            # Remove zero time entries and adjust sizes accordingly
            new_times = []
            new_sizes = [] if sizes is not None else None
            for i, t in enumerate(times):
                if t > 0.005:
                    new_times.append(t)
                    if new_sizes is not None:
                        new_sizes.append(sizes[i])
                elif new_sizes is not None and new_sizes:
                    # Add size to the last valid entry
                    new_sizes[-1] += sizes[i] if sizes is not None else 0

            # Update the DataFrame with the new lists
            df_copy.at[index, 'time_diffs'] = new_times
            if new_sizes is not None:
                df_copy.at[index, 'data_lengths'] = new_sizes
        
        PrintUtils.end_stage()
        return df_copy


class PacketBatchingMitigation(BaseMitigation):
    """
    Mitigation that batches consecutive packets together.

    Combines N consecutive packets into a single packet by summing their
    inter-packet time differences and data lengths. This reduces the sequence
    length and alters the timing and size patterns.
    """
    def __init__(self, batch_size: int = 2):
        """
        Initializes the packet batching mitigation.

        Args:
            batch_size: The number of consecutive packets to combine into one.
                        Must be an integer >= 2. Defaults to 2.
        """
        if not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("Batch size must be an integer greater than or equal to 2.")
        self.batch_size = batch_size
        PrintUtils.print_extra(f"Initialized PacketBatchingMitigation with batch_size={self.batch_size}")

    def _remove_zero_time_entries(self, times: list, sizes: list) -> tuple[list, list]:
        """
        Removes entries with zero time differences and combines their sizes
        with the previous entry. This should be applied before batching since
        packets with zero time differences arrive simultaneously.
        
        Args:
            times: List of inter-packet time differences.
            sizes: List of data lengths.
            
        Returns:
            Tuple of cleaned time and size lists.
        """
        if not times or len(times) != len(sizes):
            return times, sizes
            
        cleaned_times = []
        cleaned_sizes = []
        
        for i, (t, s) in enumerate(zip(times, sizes)):
            # Convert to numeric and validate
            try:
                time_val = float(t) if isinstance(t, (int, float)) and not math.isnan(t) else 0.0
                size_val = int(s) if isinstance(s, (int, float)) and not math.isnan(s) else 0
            except (ValueError, TypeError):
                time_val, size_val = 0.0, 0
                
            # If time is significant (> 0.005), keep as separate entry
            if time_val > 0.005:
                cleaned_times.append(time_val)
                cleaned_sizes.append(size_val)
            # If time is near zero and we have previous entries, combine with last entry
            elif cleaned_times:
                cleaned_sizes[-1] += size_val
                # Don't add to cleaned_times since this packet arrived simultaneously
            # If it's the first packet and has zero time, still keep it
            else:
                cleaned_times.append(max(0.0, time_val))
                cleaned_sizes.append(size_val)
                
        return cleaned_times, cleaned_sizes

    def _apply_batching(self, original_times: list, original_sizes: list) -> tuple[list, list]:
        """
        Applies batching logic to a single pair of time and size sequences.
        First removes zero-time entries (simultaneous packets), then applies
        regular batching.

        Args:
            original_times: List of original inter-packet time differences.
            original_sizes: List of original data lengths.

        Returns:
            A tuple containing the new list of batched time differences and the
            new list of batched data lengths.
        """
        # --- Input Validation ---
        if not isinstance(original_times, list) or not isinstance(original_sizes, list):
            PrintUtils.print_extra("PacketBatchingMitigation received non-list input.")
            return original_times, original_sizes
        if len(original_times) != len(original_sizes):
            PrintUtils.print_extra("PacketBatchingMitigation requires time/size lists of equal length.")
            return original_times, original_sizes
        if not original_times:
            return [], []

        # --- Step 1: Remove zero-time entries (simultaneous packets) ---
        cleaned_times, cleaned_sizes = self._remove_zero_time_entries(original_times, original_sizes)
        
        if not cleaned_times:
            return [], []

        # --- Step 2: Apply regular batching to cleaned data ---
        n = len(cleaned_times)
        new_times = []
        new_sizes = []

        for i in range(0, n, self.batch_size):
            # Define the slice for the current batch
            start_index = i
            end_index = min(i + self.batch_size, n) # Handle potential partial batch at the end

            # Extract the batch data
            time_batch = cleaned_times[start_index:end_index]
            size_batch = cleaned_sizes[start_index:end_index]

            # Sum the time differences in the batch
            batched_time = sum(time_batch)
            # Sum the data lengths in the batch
            batched_size = sum(size_batch)

            # Append the batched results
            new_times.append(max(0.0, batched_time)) # Ensure non-negative time
            new_sizes.append(int(batched_size))

        return new_times, new_sizes

    def transform(self, df: pd.DataFrame, median_time_delta: float = None, median_data_size: float = None) -> pd.DataFrame:
        """
        Applies packet batching to the 'time_diffs' and 'data_lengths' columns.

        Args:
            df: The input DataFrame. Must contain 'time_diffs' and 'data_lengths'
                columns with list-like entries.
            median_time_delta: Pre-calculated median time delta from the full dataset.
            median_data_size: Pre-calculated median data size from the full dataset.

        Returns:
            A new DataFrame with the packet batching mitigation applied.
        """
        mitigation_name = self.__class__.__name__
        PrintUtils.start_stage(f"Applying {mitigation_name} (batch_size={self.batch_size})")
        df_copy = df.copy()

        # --- Column Checks ---
        if 'time_diffs' not in df_copy.columns or 'data_lengths' not in df_copy.columns:
            PrintUtils.print_extra(f"Columns 'time_diffs' and 'data_lengths' required. Skipping {mitigation_name}.")
            PrintUtils.end_stage()
            return df_copy

        # --- Ensure List Format ---
        try:
            df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])
            df_copy['data_lengths'] = self._ensure_list_format(df_copy['data_lengths'])
        except Exception as e:
            PrintUtils.print_error(f"Error ensuring list format during {mitigation_name}: {e}. Skipping mitigation.")
            PrintUtils.end_stage(fail_message="List format conversion failed.")
            return df # Return original df on format error

        # --- Apply Mitigation Row-wise ---
        new_times_list = []
        new_sizes_list = []
        for index, row in df_copy.iterrows():
            original_times = row['time_diffs']
            original_sizes = row['data_lengths']

            try:
                new_t, new_s = self._apply_batching(original_times, original_sizes)
            except Exception as e:
                PrintUtils.print_extra(f"Error applying batching logic for row {index}: {e}. Skipping row modification.")
                new_t, new_s = original_times, original_sizes # Keep original on error

            new_times_list.append(new_t)
            new_sizes_list.append(new_s)

        # --- Update DataFrame Columns ---
        df_copy['time_diffs'] = new_times_list
        df_copy['data_lengths'] = new_sizes_list

        PrintUtils.end_stage()
        return df_copy

# --- Mitigation Factory ---

# Dictionary to map mitigation type strings to classes
_MITIGATION_CLASSES = {
    "RemoveZeroTimeEntries": RemoveZeroTimeEntries,
    "PacketInjectionMitigation": PacketInjectionMitigation,
    "PacketBatchingMitigation": PacketBatchingMitigation,
    # Add other mitigation classes here as they are created
}

def create_mitigation(mitigation_config: dict) -> BaseMitigation:
    """
    Factory function to create mitigation instances from configuration.

    Args:
        mitigation_config: A dictionary containing 'type' (the class name)
                           and 'params' (a dictionary of parameters for __init__).

    Returns:
        An instance of the specified mitigation class.

    Raises:
        ValueError: If the mitigation type is unknown or config is invalid.
    """
    mitigation_type = mitigation_config.get("type")
    params = mitigation_config.get("params", {})

    if not mitigation_type:
        raise ValueError("Mitigation configuration must include a 'type'.")

    if mitigation_type not in _MITIGATION_CLASSES:
        raise ValueError(f"Unknown mitigation type: {mitigation_type}. Available: {list(_MITIGATION_CLASSES.keys())}")

    MitigationClass = _MITIGATION_CLASSES[mitigation_type]

    try:
        # Instantiate with parameters
        mitigation_instance = MitigationClass(**params)
        return mitigation_instance
    except TypeError as e:
         raise ValueError(f"Error initializing mitigation {mitigation_type} with params {params}. Check parameter names and types. Error: {e}")
    except Exception as e:
        # Catch other potential errors during initialization
        raise ValueError(f"Unexpected error initializing mitigation {mitigation_type}: {e}")


def apply_mitigation_set(df: pd.DataFrame, mitigation_configs: list) -> pd.DataFrame:
    """
    Applies a sequence of mitigations to a DataFrame.

    Args:
        df: The input DataFrame.
        mitigation_configs: A list of mitigation configuration dictionaries.
                            Mitigations are applied in the order they appear
                            in this list.
        median_time_delta: Pre-calculated median time delta from the full dataset.
        median_data_size: Pre-calculated median data size from the full dataset.

    Returns:
        The DataFrame after all mitigations have been applied.
    """
    df_mitigated = df # Start with the original DataFrame
    if not mitigation_configs:
        PrintUtils.print_extra("No mitigations specified for this set.")
        return df_mitigated

    PrintUtils.start_stage(f"Applying mitigation set ({len(mitigation_configs)} mitigation(s))")
    for i, config in enumerate(mitigation_configs):
        try:
            mitigation = create_mitigation(config)
            PrintUtils.print_extra(f"Applying step {i+1}/{len(mitigation_configs)}: {mitigation.__class__.__name__}")
            df_mitigated = mitigation.transform(df_mitigated)
            # Optional: Add checks here if needed (e.g., check if columns still exist)
            if df_mitigated is None or not isinstance(df_mitigated, pd.DataFrame):
                 raise RuntimeError(f"Mitigation {mitigation.__class__.__name__} returned invalid result.")
        except (ValueError, RuntimeError) as e:
            PrintUtils.print_error(f"Error applying mitigation step {i+1} ({config.get('type', 'Unknown')}): {e}")
            PrintUtils.end_stage(fail_message="Mitigation set application failed.")
            # Depending on desired behavior, you might return original df or raise further
            return df # Return the DataFrame as it was before the failing step
        except Exception as e:
             PrintUtils.print_error(f"Unexpected error during mitigation step {i+1} ({config.get('type', 'Unknown')}): {e}")
             PrintUtils.end_stage(fail_message="Mitigation set application failed due to unexpected error.")
             import traceback
             traceback.print_exc()
             return df # Return the DataFrame as it was before the failing step


    PrintUtils.end_stage() # End stage for the whole set
    return df_mitigated
