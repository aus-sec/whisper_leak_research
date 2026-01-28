# bert_time_series_classifier.py

import ast
import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
import warnings

# Suppress excessive warnings from transformers library (optional)
hf_logging.set_verbosity_error() 

# Assuming base_classifier.py is in the same directory or accessible via python path
from .base_classifier import BaseClassifier 

class BERTTimeSeriesClassifier(BaseClassifier):
    """
    A classifier using a pre-trained BERT model adapted for time series data.
    It buckets normalized time_diffs and data_lengths, converts them to new tokens,
    combines them based on flags (interleave/single_feature), and feeds them 
    into the transformer model for binary classification.
    """

    def __init__(self, 
                 normalization_params,
                 # Bucket boundaries for *normalized* data
                 time_boundaries_norm, 
                 len_boundaries_norm,
                 num_buckets=50,
                 # Pre-trained model config
                 model_name='distilbert-base-uncased', # Good balance of size/performance
                 # Classifier head config
                 fc_dims=[128, 64], 
                 dropout_rate=0.3,
                 single_feature=False,
                 interleave=False # If True and single_feature=False, alternates time/len tokens. Otherwise concatenates len then time.
                 ):
        """
        Creates an instance of the BERTTimeSeriesClassifier.

        Args:
            normalization_params (tuple): A tuple containing 
                (time_mean, time_std, size_mean, size_std, max_len).
            time_boundaries_norm (list/np.array): Quantile boundaries for normalized time_diffs. 
                                                  Should have num_buckets - 1 elements.
            len_boundaries_norm (list/np.array): Quantile boundaries for normalized data_lengths.
                                                 Should have num_buckets - 1 elements.
            num_buckets (int): The number of buckets to divide time and length data into.
            model_name (str): The name of the pre-trained model from Hugging Face Hub.
            fc_dims (list): List of dimensions for fully connected layers in the classification head.
            dropout_rate (float): Dropout rate for regularization in the classification head.
            single_feature (bool): If True, uses only data_lengths feature. If False, uses both.
            interleave (bool): If True and single_feature=False, alternates time and length tokens. 
                               If False and single_feature=False, concatenates length tokens then time tokens.
                               Ignored if single_feature=True.
        """
        super().__init__(normalization_params)
        self.class_name = self.__class__.__name__

        # --- Parameter Validation ---
        if not (isinstance(time_boundaries_norm, (list, np.ndarray)) and \
                len(time_boundaries_norm) == num_buckets - 1):
             raise ValueError(f"time_boundaries_norm must be a list/array of length num_buckets-1 ({num_buckets-1})")
        if not (isinstance(len_boundaries_norm, (list, np.ndarray)) and \
                len(len_boundaries_norm) == num_buckets - 1):
             raise ValueError(f"len_boundaries_norm must be a list/array of length num_buckets-1 ({num_buckets-1})")

        # --- Store Config ---
        self.time_boundaries_norm = np.array(time_boundaries_norm)
        self.len_boundaries_norm = np.array(len_boundaries_norm)
        self.num_buckets = num_buckets
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.fc_dims = fc_dims
        self.single_feature = single_feature
        self.interleave = interleave
        
        # Original max sequence length (from normalization params)
        self.original_max_len = self.max_len
        
        # --- MODIFIED: Correct calculation of bert_max_len ---
        # Max length for BERT input depends on feature combination + special tokens ([CLS] + [SEP])
        if self.single_feature:
            # Only one feature type (length)
            self.bert_max_len = self.original_max_len + 2 
        else:
            # Both features are used (either interleaved or concatenated)
            # Max sequence length = original_max_len (time) + original_max_len (length)
            self.bert_max_len = (2 * self.original_max_len) + 2
        # --- END MODIFICATION ---
        
        if self.bert_max_len > 512:
            # Most standard BERT models have a 512 token limit
            original_calculated_len = self.bert_max_len
            self.bert_max_len = 512
            # Adjust original_max_len proportionally to fit within 512 BERT limit
            if self.single_feature:
                self.original_max_len = self.bert_max_len - 2
            else:
                self.original_max_len = (self.bert_max_len - 2) // 2 # Floor division
            print(f"Warning: Calculated BERT input length ({original_calculated_len}) exceeds 512.")
            print(f"         Reducing BERT max length to {self.bert_max_len}.")
            print(f"         Effective original sequence max length reduced to {self.original_max_len}.")

        
        self.args = {
            'model_name': self.model_name,
            'num_buckets': self.num_buckets,
            # Store boundaries as lists for JSON serialization
            'time_boundaries_norm': self.time_boundaries_norm.tolist(), 
            'len_boundaries_norm': self.len_boundaries_norm.tolist(),
            'fc_dims': self.fc_dims,
            'dropout_rate': self.dropout_rate,
            'single_feature': self.single_feature, # Store flag
            'interleave': self.interleave,       # Store flag
            # Note: normalization_params are handled by the BaseClassifier save/load
        }

        # --- Load Tokenizer and Model ---
        # Use cache_dir to potentially control download location
        # cache_dir = "./hf_cache" 
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # self.bert_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Default behavior: downloads to ~/.cache/huggingface/hub
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model: {model_name}")
        self.bert_model = AutoModel.from_pretrained(model_name)

        # --- Add New Tokens ---
        self.len_tokens = [f"[LEN_{i}]" for i in range(num_buckets)]
        self.time_tokens = [f"[TIME_{i}]" for i in range(num_buckets)]
        special_tokens_to_add = self.len_tokens + self.time_tokens
        
        num_added_toks = self.tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
        print(f"Added {num_added_toks} new tokens to tokenizer.")
        
        # --- Resize Model Embeddings ---
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        print(f"Resized model embeddings to: {len(self.tokenizer)}")

        # --- Get Token IDs ---
        self.len_token_ids = self.tokenizer.convert_tokens_to_ids(self.len_tokens)
        self.time_token_ids = self.tokenizer.convert_tokens_to_ids(self.time_tokens)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        if self.pad_token_id is None:
             # Some models might not have a default pad token (like GPT2)
             # Add one if missing, although DistilBERT should have it.
             print("Warning: Tokenizer has no default pad token. Adding '[PAD]'.")
             # Use add_special_tokens to handle potential duplicates gracefully
             self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             self.pad_token_id = self.tokenizer.pad_token_id
             self.bert_model.resize_token_embeddings(len(self.tokenizer))
             print(f"Resized model embeddings after adding pad token: {len(self.tokenizer)}")


        # --- Classification Head ---
        bert_output_dim = self.bert_model.config.hidden_size
        
        fc_layers = []
        input_dim = bert_output_dim
        
        for dim in fc_dims:
            fc_layers.extend([
                nn.Linear(input_dim, dim),
                # Consider BatchNorm or LayerNorm depending on performance
                nn.LayerNorm(dim), 
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = dim
            
        # Final output layer
        fc_layers.append(nn.Linear(input_dim, 1))
        
        # Combine all FC layers
        self.fc_layers = nn.Sequential(*fc_layers)

        # Initialize weights for the classification head
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize weights for the classification head."""
        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                 nn.init.constant_(m.weight, 1.0)
                 nn.init.constant_(m.bias, 0.0)

    def _bucket_and_tokenize_sample(self, norm_time_vals, norm_size_vals):
        """
        Buckets, tokenizes, combines (interleaving or concatenating based on flags), 
        and pads a single sample.
        """
        
        # 1. Determine actual sequence length
        # Assumes input arrays norm_time_vals, norm_size_vals are already trimmed to original length
        seq_len = len(norm_time_vals) # Should be <= self.original_max_len

        # 2. Bucket the normalized values
        time_bucket_indices = np.digitize(norm_time_vals, self.time_boundaries_norm)
        len_bucket_indices = np.digitize(norm_size_vals, self.len_boundaries_norm)

        # 3. Map bucket indices to token IDs
        time_tok_ids = [self.time_token_ids[idx] for idx in time_bucket_indices]
        len_tok_ids = [self.len_token_ids[idx] for idx in len_bucket_indices]

        # --- MODIFIED: Implement interleave logic ---
        # 4. Combine/Interleave tokens based on flags
        if self.single_feature:
            # Use only length tokens if single_feature is True
            combined_tokens = len_tok_ids 
        else:
            # Use both features
            if self.interleave:
                # Alternate time and length tokens: [T0, L0, T1, L1, ...]
                combined_tokens = []
                for i in range(seq_len):
                    combined_tokens.append(time_tok_ids[i]) # Time first
                    combined_tokens.append(len_tok_ids[i]) # Then Length
            else:
                # Concatenate: all length tokens, then all time tokens: [L0,...,Ln, T0,...,Tn]
                combined_tokens = len_tok_ids + time_tok_ids 
        # --- END MODIFICATION ---

        # 5. Add special tokens ([CLS] at start, [SEP] at end)
        # Use combined_tokens which now holds the correctly ordered sequence
        final_token_ids = [self.cls_token_id] + combined_tokens + [self.sep_token_id]
        
        # 6. Pad sequence to self.bert_max_len
        current_len = len(final_token_ids)
        padding_len = self.bert_max_len - current_len
        
        if padding_len < 0:
            # If combined sequence exceeds max length, truncate
            # Truncate from the end before adding [SEP]
            final_token_ids = final_token_ids[:self.bert_max_len-1] + [self.sep_token_id]
            attention_mask = [1] * self.bert_max_len
            # Optional: Log truncation event
            # print(f"Warning: Sequence truncated from {current_len} to {self.bert_max_len}")
        else:
            # Pad with pad_token_id
            final_token_ids = final_token_ids + ([self.pad_token_id] * padding_len)
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = ([1] * current_len) + ([0] * padding_len)
            
        # Ensure final length matches bert_max_len exactly
        if len(final_token_ids) != self.bert_max_len:
             # This should ideally not happen with the logic above, but as a safeguard:
             warnings.warn(f"Length mismatch after padding/truncation: expected {self.bert_max_len}, got {len(final_token_ids)}. Check logic.")
             # Force length (less ideal fix, investigate root cause if this occurs)
             final_token_ids = final_token_ids[:self.bert_max_len]
             attention_mask = attention_mask[:self.bert_max_len]
             if len(final_token_ids) < self.bert_max_len:
                 pad_needed = self.bert_max_len - len(final_token_ids)
                 final_token_ids.extend([self.pad_token_id] * pad_needed)
                 attention_mask.extend([0] * pad_needed)


        return final_token_ids, attention_mask

    def forward(self, x):
        """
        Implements the forward pass: bucketing, tokenization, BERT, classification head.
        
        Args:
            x (torch.Tensor): Input tensor from Loader of shape 
                              [batch_size, 2, max_len_from_loader], containing
                              *normalized* and *padded* time_diffs and data_lengths.
                              x[:, 0, :] = normalized time_diffs
                              x[:, 1, :] = normalized data_lengths
                              Note: max_len_from_loader should ideally match self.original_max_len
                              after the potential adjustment in __init__.
        """
        batch_size = x.size(0)
        loader_max_len = x.size(2)
        device = x.device

        all_input_ids = []
        all_attention_masks = []

        # Process each sample in the batch
        for i in range(batch_size):
            # Extract normalized, padded sequences for the i-th sample
            # Use .detach() in case gradients are attached unexpectedly
            norm_time_padded = x[i, 0, :].detach().cpu().numpy()
            norm_size_padded = x[i, 1, :].detach().cpu().numpy()

            # --- Find original length before padding ---
            # This relies on the assumption that the dataloader uses 0 for padding *after* normalization.
            # A more robust approach is if the dataloader provides sequence lengths or masks.
            
            # Find the indices of the last non-zero element. Max length if all non-zero.
            non_zero_time_idx = np.where(norm_time_padded != 0)[0]
            non_zero_size_idx = np.where(norm_size_padded != 0)[0]
            
            # Determine effective sequence length, capped by self.original_max_len
            effective_len = 0
            if len(non_zero_time_idx) > 0:
                effective_len = max(effective_len, non_zero_time_idx[-1] + 1)
            if len(non_zero_size_idx) > 0:
                effective_len = max(effective_len, non_zero_size_idx[-1] + 1)
            
            # Ensure the effective length does not exceed the model's configured original_max_len
            effective_len = min(effective_len, self.original_max_len)
                
            if effective_len == 0:
                 # Handle potentially empty sequences after padding removal/length capping
                 # Create a minimal sequence [CLS] [SEP] + padding
                 input_ids = [self.cls_token_id, self.sep_token_id] + [self.pad_token_id] * (self.bert_max_len - 2)
                 attention_mask = [1, 1] + [0] * (self.bert_max_len - 2)
                 
            else:
                # Extract the actual data up to the determined effective length
                norm_time_vals = norm_time_padded[:effective_len]
                norm_size_vals = norm_size_padded[:effective_len]

                # Bucket, tokenize, combine, pad for this sample
                input_ids, attention_mask = self._bucket_and_tokenize_sample(norm_time_vals, norm_size_vals)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # Convert lists to tensors
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long).to(device)
        
        # --- Sanity Check Tensor Shapes ---
        if input_ids_tensor.shape != (batch_size, self.bert_max_len):
            warnings.warn(f"Input IDs tensor shape mismatch: Expected ({batch_size}, {self.bert_max_len}), Got {input_ids_tensor.shape}. Check tokenization/padding.")
        if attention_mask_tensor.shape != (batch_size, self.bert_max_len):
            warnings.warn(f"Attention mask tensor shape mismatch: Expected ({batch_size}, {self.bert_max_len}), Got {attention_mask_tensor.shape}. Check tokenization/padding.")


        # --- Pass through BERT model ---
        outputs = self.bert_model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor
        )

        # --- Get representation for classification ---
        # Use the output of the [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :] 
        # Alternative: Pool the sequence outputs (e.g., mean pooling)
        # sequence_output = outputs.last_hidden_state
        # pooled_output = torch.mean(sequence_output * attention_mask_tensor.unsqueeze(-1), dim=1)

        # --- Pass through classification head ---
        logits = self.fc_layers(cls_output) # Shape: [batch_size, 1]
        
        return logits # Return shape [batch_size, 1]
    
    def get_optimizer_params(self, lr):
        # Differential learning rates
        optimizer_grouped_parameters = [
            {'params': self.bert_model.parameters(), 'lr': lr}, # Lower LR for BERT base
            {'params': self.fc_layers.parameters(), 'lr': lr * 100} # 100x higher LR for classifier head
        ]
        # You might need to adjust the multiplier (e.g., 10, 50, 100, 200)
        print(f"Using differential learning rates: Base={lr}, Head={lr*100}")
        return optimizer_grouped_parameters

    # Note: calculate_boundaries static method remains unchanged
    @staticmethod
    def calculate_boundaries(df, num_buckets, norm):
        """
        Calculates quantile boundaries for normalized time_diffs and data_lengths.

        This method processes a DataFrame, normalizes the specified columns using
        the provided parameters, and computes quantile boundaries suitable for 
        bucketing in the BERTTimeSeriesClassifier.

        Args:
            df (pd.DataFrame): DataFrame containing 'time_diffs' and 'data_lengths' columns.
                               Values can be lists or string representations of lists.
            num_buckets (int): The number of buckets to create (N). This method will
                               calculate N-1 quantile boundaries.
            norm (dict): A dictionary containing the pre-calculated normalization
                         parameters: {"time_mean": ..., "time_std": ..., 
                                      "size_mean": ..., "size_std": ...}.

        Returns:
            tuple: A tuple containing two numpy arrays:
                   (time_boundaries_norm, len_boundaries_norm).
                   Each array contains num_buckets - 1 boundary values.

        Raises:
            ValueError: If columns 'time_diffs' or 'data_lengths' are missing,
                        if num_buckets is less than 2, or if normalization_params
                        are invalid or missing required keys.
            TypeError: If normalization_params is not a dictionary.
        """
        # --- Input Validation ---
        if 'time_diffs' not in df.columns or 'data_lengths' not in df.columns:
            raise ValueError("DataFrame must contain 'time_diffs' and 'data_lengths' columns.")
        if not isinstance(num_buckets, int) or num_buckets < 2:
             raise ValueError("num_buckets must be an integer greater than or equal to 2.")
        if not isinstance(norm, (dict)):
            raise TypeError("Normalization parameters 'norm' must be a dictionary.")
        required_keys = ["time_mean", "time_std", "size_mean", "size_std"]
        if not all(key in norm for key in required_keys):
            raise ValueError(f"Normalization parameters must include keys: {required_keys}.")
        
        time_mean = norm["time_mean"]
        time_std = norm["time_std"]
        size_mean = norm["size_mean"]
        size_std = norm["size_std"]

        # Ensure standard deviations are positive for safe division
        # Use small epsilon to prevent division by zero if std dev is exactly 0
        epsilon = 1e-8 
        time_std_safe = time_std if time_std > 0 else epsilon
        size_std_safe = size_std if size_std > 0 else epsilon

        # --- Data Preparation and Normalization ---
        all_norm_time_vals = []
        all_norm_size_vals = []

        for idx, row in df.iterrows(): # Use idx for better error reporting
            # Handle string representations safely
            try:
                time_vals = ast.literal_eval(row['time_diffs']) if isinstance(row['time_diffs'], str) else row['time_diffs']
                size_vals = ast.literal_eval(row['data_lengths']) if isinstance(row['data_lengths'], str) else row['data_lengths']
            except (ValueError, SyntaxError) as e:
                 print(f"Warning: Skipping row index {idx} due to parsing error: {e}. Row content:\n{row['time_diffs']}\n{row['data_lengths']}")
                 continue # Skip rows that cannot be parsed

            if not isinstance(time_vals, list) or not isinstance(size_vals, list):
                 print(f"Warning: Skipping row index {idx} due to unexpected data type. Expected lists. Got: {type(time_vals)}, {type(size_vals)}")
                 continue # Skip rows with unexpected types
            
            if len(time_vals) != len(size_vals):
                print(f"Warning: Skipping row index {idx} due to mismatched lengths between time_diffs ({len(time_vals)}) and data_lengths ({len(size_vals)}).")
                continue # Skip rows with mismatched lengths

            # Normalize and add to flattened lists (only non-empty lists)
            if time_vals: # Check if lists are not empty
               all_norm_time_vals.extend([(float(val) - time_mean) / time_std_safe for val in time_vals])
            if size_vals:
               all_norm_size_vals.extend([(float(val) - size_mean) / size_std_safe for val in size_vals])


        if not all_norm_time_vals or not all_norm_size_vals:
             # Check if *any* data was processed, even if one list ended up empty
             if not df.empty and not (all_norm_time_vals or all_norm_size_vals):
                 raise ValueError("After processing, no valid numeric time or size values were found in the DataFrame. Check data format and content.")
             elif not all_norm_time_vals:
                 warnings.warn("No valid time values found after processing. Time boundaries cannot be calculated.")
                 time_boundaries_norm = np.array([]) # Or handle as error depending on requirements
             elif not all_norm_size_vals:
                 warnings.warn("No valid size values found after processing. Length boundaries cannot be calculated.")
                 len_boundaries_norm = np.array([]) # Or handle as error

        # --- Quantile Calculation ---
        # Calculate N-1 quantiles to define N buckets
        # Quantiles range from 1/N to (N-1)/N
        quantiles = np.linspace(0, 1, num_buckets + 1)[1:-1] 
        
        # Use np.quantile on the *normalized* flattened data if available
        if all_norm_time_vals:
            time_boundaries_norm = np.quantile(all_norm_time_vals, quantiles)
            # Ensure boundaries are unique 
            if len(np.unique(time_boundaries_norm)) != num_buckets - 1:
                 print(f"Warning: Duplicate time boundaries found ({len(np.unique(time_boundaries_norm))} unique values for {num_buckets-1} boundaries). This might occur with skewed data or low num_buckets. Consider adjusting num_buckets or data preprocessing.")
        else:
             time_boundaries_norm = np.array([]) # Return empty if no data

        if all_norm_size_vals:
            len_boundaries_norm = np.quantile(all_norm_size_vals, quantiles)
             # Ensure boundaries are unique
            if len(np.unique(len_boundaries_norm)) != num_buckets - 1:
                 print(f"Warning: Duplicate length boundaries found ({len(np.unique(len_boundaries_norm))} unique values for {num_buckets-1} boundaries). This might occur with skewed data or low num_buckets. Consider adjusting num_buckets or data preprocessing.")
        else:
            len_boundaries_norm = np.array([]) # Return empty if no data

        return time_boundaries_norm, len_boundaries_norm

