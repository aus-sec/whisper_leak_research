# core/classifiers/combined_lstm_bert_classifier.py

import math
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel, logging as hf_logging

# Suppress excessive warnings from transformers library (optional)
hf_logging.set_verbosity_error()
from core.utils import PrintUtils

# Assuming BaseClassifier is in the parent directory or accessible path
from .base_classifier import BaseClassifier

class CombinedLSTMBERTClassifier(BaseClassifier):
    """
    A combined classifier leveraging both an LSTM network (on time and size)
    and a BERT-based network (on size only) for time series classification.

    The LSTM component processes both normalized time and size sequences using
    bidirectional LSTMs with an attention mechanism.

    The BERT component processes only the normalized size sequence. It represents
    each size value using a special '[SIZE_VAL]' token and adds a learned
    linear projection of the continuous size value to the token's embedding
    before feeding it through a pre-trained BERT encoder.

    The outputs from the LSTM's attention layer and the BERT's [CLS] token
    are concatenated and passed through a final classification head.
    """

    def __init__(self,
                 normalization_params, # Inherited from BaseClassifier, contains 'max_len', means, stds
                 # --- LSTM Configuration ---
                 lstm_hidden_size=128,
                 lstm_num_layers=2,
                 lstm_dropout_rate=0.3,
                 lstm_embedding_dim=32,
                 lstm_bidirectional=True,
                 lstm_attention_dim=128,
                 # --- BERT Configuration ---
                 freeze_bert=False,  # New parameter to control freezing
                 bert_lr_factor=0.1,  # New parameter: BERT learning rate = main_lr * factor
                 bert_model_name='distilbert-base-uncased', # Or another suitable BERT model
                 # --- Shared Classifier Head Config ---
                 fc_dims=[128, 64],
                 dropout_rate=0.3
                 ):
        """
        Initializes the CombinedLSTMBERTClassifier.

        Args:
            normalization_params (dict): Dictionary containing normalization parameters:
                                         'time_mean', 'time_std', 'size_mean', 'size_std', 'max_len'.
                                         Provided by the Loader and stored in self.norm by BaseClassifier.
            lstm_hidden_size (int): Size of LSTM hidden layers.
            lstm_num_layers (int): Number of LSTM layers.
            lstm_dropout_rate (float): Dropout rate for LSTM layers (if num_layers > 1).
            lstm_embedding_dim (int): Dimension for the input embeddings before LSTM.
            lstm_bidirectional (bool): Whether the LSTM layers are bidirectional.
            lstm_attention_dim (int): Dimension of the attention layer applied to LSTM outputs.
            bert_model_name (str): Name of the pre-trained BERT model from Hugging Face Hub.
            fc_dims (list): List of dimensions for the fully connected layers in the final head.
            dropout_rate (float): Dropout rate for regularization in the final FC head.
        """
        super().__init__(normalization_params) # Stores norm_params in self.norm, including self.max_len
        self.class_name = self.__class__.__name__

        # --- Store configuration parameters ---
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout_rate = lstm_dropout_rate
        self.lstm_embedding_dim = lstm_embedding_dim
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_attention_dim = lstm_attention_dim
        self.lstm_direction_factor = 2 if lstm_bidirectional else 1

        self.freeze_bert = freeze_bert
        self.bert_lr_factor = bert_lr_factor
        self.bert_model_name = bert_model_name

        self.fc_dims = fc_dims
        self.dropout_rate = dropout_rate # Used in the final classifier head

        # --- Store args for saving/loading ---
        self.args = {
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'lstm_dropout_rate': self.lstm_dropout_rate,
            'lstm_embedding_dim': self.lstm_embedding_dim,
            'lstm_bidirectional': self.lstm_bidirectional,
            'lstm_attention_dim': self.lstm_attention_dim,
            'bert_model_name': self.bert_model_name,
            'fc_dims': self.fc_dims,
            'dropout_rate': self.dropout_rate,
            'freeze_bert': self.freeze_bert,
            'bert_lr_factor': self.bert_lr_factor,
            # normalization_params are handled by BaseClassifier save/load
        }

        # ======================== LSTM Component Initialization ========================
        # Embedding layers for time and size inputs (each is 1D -> embedding_dim)
        self.lstm_time_embedding = nn.Sequential(
            nn.Linear(1, self.lstm_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.lstm_embedding_dim)
        )
        self.lstm_size_embedding = nn.Sequential(
            nn.Linear(1, self.lstm_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.lstm_embedding_dim)
        )

        # Combined feature dimension after embedding
        self.lstm_feature_dim = self.lstm_embedding_dim * 2

        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_feature_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=self.lstm_bidirectional,
            dropout=self.lstm_dropout_rate if self.lstm_num_layers > 1 else 0
        )

        # Attention mechanism for LSTM output
        self.lstm_attention = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * self.lstm_direction_factor, self.lstm_attention_dim),
            nn.Tanh(),
            nn.Linear(self.lstm_attention_dim, 1)
        )
        # ===================== End LSTM Component Initialization =====================


        # ======================== BERT Component Initialization ========================
        # --- Load Tokenizer and Base BERT Model ---
        PrintUtils.print_extra(f"Loading BERT tokenizer: {self.bert_model_name}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        PrintUtils.print_extra(f"Loading BERT model: {self.bert_model_name}")
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
        self.bert_hidden_size = self.bert_model.config.hidden_size

        # --- Add Special Token for Size Values ---
        self.size_val_token = "[SIZE_VAL]"
        special_tokens_to_add = [self.size_val_token]
        num_added_toks = self.bert_tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
        PrintUtils.print_extra(f"Added {num_added_toks} special token(s) to BERT tokenizer: {special_tokens_to_add}")

        # --- Resize Model Embeddings to accommodate new token ---
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        PrintUtils.print_extra(f"Resized BERT model embeddings to: {len(self.bert_tokenizer)}")

        # --- Get Token IDs (after adding new tokens) ---
        self.size_val_token_id = self.bert_tokenizer.convert_tokens_to_ids(self.size_val_token)
        self.cls_token_id = self.bert_tokenizer.cls_token_id
        self.sep_token_id = self.bert_tokenizer.sep_token_id
        self.pad_token_id = self.bert_tokenizer.pad_token_id
        if self.pad_token_id is None:
             PrintUtils.print_extra("Warning: BERT Tokenizer has no default pad token. Adding '[PAD]'.")
             self.bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             self.pad_token_id = self.bert_tokenizer.pad_token_id
             self.bert_model.resize_token_embeddings(len(self.bert_tokenizer)) # Resize again if PAD added

        # --- Determine BERT Input Sequence Length ---
        # Max sequence length for the base BERT model
        self.bert_model_max_native_len = self.bert_model.config.max_position_embeddings
        # Max number of [SIZE_VAL] tokens based on input data's max_len
        self.input_data_max_len = self.norm['max_len']
        # Calculated BERT input length: [CLS] + N * [SIZE_VAL] + [SEP]
        self.bert_calculated_input_len = 1 + self.input_data_max_len + 1

        if self.bert_calculated_input_len > self.bert_model_max_native_len:
            # Adjust the number of size values we can actually use to fit BERT
            self.bert_effective_max_len_for_size = self.bert_model_max_native_len - 2 # Account for [CLS], [SEP]
            PrintUtils.print_extra(f"Warning: BERT model max input length ({self.bert_model_max_native_len}) "
                             f"exceeded. Adjusting effective max size values to {self.bert_effective_max_len_for_size}.")
            self.bert_input_seq_len = self.bert_model_max_native_len
        else:
            self.bert_effective_max_len_for_size = self.input_data_max_len
            self.bert_input_seq_len = self.bert_calculated_input_len
            # Pad sequence to this length if shorter

        PrintUtils.print_extra(f"Effective max number of size values used by BERT: {self.bert_effective_max_len_for_size}")
        PrintUtils.print_extra(f"Resulting BERT input sequence length (incl. CLS/SEP/PAD): {self.bert_input_seq_len}")

        # --- Linear Projection Layer for Continuous Size Values ---
        self.bert_size_embed_proj = nn.Linear(1, self.bert_hidden_size)

        PrintUtils.print_extra(f"Created projection layer (1 -> {self.bert_hidden_size}) for size values.")
        # ===================== End BERT Component Initialization =====================


        # ======================== Final Classifier Head ========================
        # Input dimension to the FC layers is the concatenation of LSTM and BERT outputs
        combined_input_dim = (self.lstm_hidden_size * self.lstm_direction_factor) + self.bert_hidden_size
        PrintUtils.print_extra(f"Combined input dimension for final classifier head: {combined_input_dim}")

        fc_layers_list = []
        current_dim = combined_input_dim

        for dim in self.fc_dims:
            fc_layers_list.extend([
                nn.Linear(current_dim, dim),
                # Using LayerNorm as it's often preferred over BatchNorm with sequential data features
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate) # Use the shared dropout rate
            ])
            current_dim = dim

        # Final output layer producing logits for binary classification
        fc_layers_list.append(nn.Linear(current_dim, 1))

        # Combine all FC layers into a Sequential module
        self.fc_layers = nn.Sequential(*fc_layers_list)
        # ===================== End Final Classifier Head =====================

        # --- BERT Model Freezing (if requested) ---
        if self.freeze_bert:
            PrintUtils.print_extra("Freezing BERT model parameters")
            for param in self.bert_model.parameters():
                param.requires_grad = False
        else:
            PrintUtils.print_extra("BERT model parameters will be fine-tuned")
            # Optionally add parameter group information for the optimizer
            # This helps the calling code configure the optimizer with proper learning rates
            self.bert_parameters = list(self.bert_model.parameters())
            self.non_bert_parameters = (
                list(self.lstm_time_embedding.parameters()) +
                list(self.lstm_size_embedding.parameters()) +
                list(self.lstm.parameters()) +
                list(self.lstm_attention.parameters()) +
                list(self.bert_size_embed_proj.parameters()) +
                list(self.fc_layers.parameters())
            )

        # --- Initialize Weights ---
        self._init_weights()

    def get_optimizer_params(self, base_lr):
        """
        Returns parameter groups for the optimizer with appropriate learning rates.
        
        Args:
            base_lr (float): Base learning rate
            
        Returns:
            list: Parameter groups for optimizer
        """
        if hasattr(self, 'freeze_bert') and self.freeze_bert:
            # If BERT is frozen, return all trainable parameters with the same learning rate
            return [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': base_lr}]
        
        # If BERT is not frozen, use different learning rates
        bert_lr = base_lr * self.bert_lr_factor
        
        # Make sure we use the correct attribute names from __init__
        return [
            {'params': list(self.lstm_time_embedding.parameters()) +
                    list(self.lstm_size_embedding.parameters()) +
                    list(self.lstm.parameters()) +
                    list(self.lstm_attention.parameters()) +
                    list(self.bert_size_embed_proj.parameters()) +  # This matches your attribute name
                    list(self.fc_layers.parameters()),
            'lr': base_lr},
            {'params': self.bert_model.parameters(), 'lr': bert_lr}
        ]


    def _init_weights(self):
        """
        Initialize weights for LSTM, Attention, BERT Projections, and Classifier Head.
        BERT base model weights are pre-trained and loaded, so not re-initialized here.
        """
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initialize Attention weights
        for m in self.lstm_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Initialize BERT projection layer weights
        nn.init.xavier_uniform_(self.bert_size_embed_proj.weight)
        if self.bert_size_embed_proj.bias is not None:
            nn.init.constant_(self.bert_size_embed_proj.bias, 0.0)

        # Initialize final FC classification head weights
        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                 nn.init.constant_(m.weight, 1.0)
                 nn.init.constant_(m.bias, 0.0)


    def _apply_lstm_attention(self, lstm_output, mask):
        """
        Apply the attention mechanism to the LSTM output sequence.

        Args:
            lstm_output (torch.Tensor): Output from the LSTM layer
                                       Shape: [batch_size, seq_len, hidden_size * direction_factor]
            mask (torch.Tensor): Boolean mask indicating non-padded elements.
                                 Shape: [batch_size, seq_len]

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - context_vector (torch.Tensor): Weighted sum of LSTM outputs.
                                                 Shape: [batch_size, hidden_size * direction_factor]
                - attention_weights (torch.Tensor): Attention weights applied.
                                                    Shape: [batch_size, seq_len, 1]
        """
        # Calculate attention scores
        # attention_scores shape: [batch_size, seq_len, 1]
        attention_scores = self.lstm_attention(lstm_output)

        # Apply mask: Set scores for padded elements to -infinity before softmax
        # Mask needs to be unsqueezed to match attention_scores shape: [batch_size, seq_len, 1]
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(2) == 0, float('-inf'))

        # Calculate attention weights (softmax over sequence dimension)
        # attention_weights shape: [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Calculate context vector (weighted sum)
        # context_vector shape: [batch_size, hidden_size * direction_factor]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        return context_vector, attention_weights


    def forward(self, x):
        """
        Implements the forward pass for the combined classifier.

        Args:
            x (torch.Tensor): Input tensor from the Loader.
                              Shape: [batch_size, 2, input_data_max_len]
                              x[:, 0, :] = normalized time diffs (z-score)
                              x[:, 1, :] = normalized data lengths (z-score)

        Returns:
            torch.Tensor: Raw logits output from the final classification layer.
                          Shape: [batch_size, 1]
        """
        batch_size = x.size(0)
        device = x.device

        # Input sequence length based on the loader's padding (self.norm['max_len'])
        loader_padded_seq_len = x.size(2)

        # --- Create mask for padding (shared by LSTM and BERT logic) ---
        # A position is considered non-padded if *either* time or size is non-zero
        # This mask reflects the original sequence length before padding by the Loader.
        # Shape: [batch_size, loader_padded_seq_len]
        # Use absolute value sum in case normalized values are small but non-zero
        # Add a small epsilon for floating point comparisons
        original_seq_mask = (torch.abs(x[:, 0, :]) + torch.abs(x[:, 1, :]) > 1e-9).float()
        # Calculate original sequence lengths for each item in the batch
        original_lengths = original_seq_mask.sum(dim=1).cpu().int()

        # ========================== LSTM Path (Time + Size) ==========================
        # Extract time and size sequences, add channel dim for Linear layer
        # Shape: [batch_size, loader_padded_seq_len, 1]
        time_seq = x[:, 0, :].unsqueeze(-1)
        size_seq = x[:, 1, :].unsqueeze(-1)

        # Apply input embeddings
        # Shape: [batch_size, loader_padded_seq_len, lstm_embedding_dim]
        embedded_time = self.lstm_time_embedding(time_seq)
        embedded_size = self.lstm_size_embedding(size_seq)

        # Concatenate embedded features along the last dimension
        # Shape: [batch_size, loader_padded_seq_len, lstm_feature_dim]
        lstm_input_embedded = torch.cat((embedded_time, embedded_size), dim=2)

        # Pack the sequence for LSTM efficiency (handles variable lengths)
        # Ensure lengths are on CPU and are > 0
        packed_lengths = torch.clamp(original_lengths, min=1) # Clamp min length to 1 for packing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            lstm_input_embedded,
            lengths=packed_lengths.cpu(), # Needs lengths on CPU
            batch_first=True,
            enforce_sorted=False # Data is not pre-sorted by length
        )

        # Pass through LSTM
        # packed_output: contains outputs for each step
        # hidden: contains final hidden and cell states
        packed_output, (lstm_hidden, lstm_cell) = self.lstm(packed_embedded)

        # Unpack the sequence
        # lstm_output shape: [batch_size, loader_padded_seq_len, lstm_hidden_size * lstm_direction_factor]
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=loader_padded_seq_len # Pad back to the original Loader length
        )

        # Apply attention mechanism
        # lstm_context_vector shape: [batch_size, lstm_hidden_size * lstm_direction_factor]
        lstm_context_vector, _ = self._apply_lstm_attention(lstm_output, original_seq_mask)
        # ======================== End LSTM Path ==========================


        # ======================== BERT Path (Size Only) ========================
        # Prepare inputs for BERT
        bert_input_ids_batch = []
        bert_size_values_batch = []
        bert_original_lengths_batch = [] # Store lengths relevant to BERT's truncated input

        # Extract normalized size values (already z-scored by Loader)
        # Shape: [batch_size, loader_padded_seq_len]
        norm_size_values = x[:, 1, :]

        for i in range(batch_size):
            # Get the original length for this sample, capped by BERT's effective max length
            current_original_len = original_lengths[i].item()
            bert_len_for_sample = min(current_original_len, self.bert_effective_max_len_for_size)
            bert_original_lengths_batch.append(bert_len_for_sample)

            # Build token ID sequence: [CLS] [SIZE_VAL]... [SEP] [PAD]...
            input_ids_sample = [self.cls_token_id]
            input_ids_sample.extend([self.size_val_token_id] * bert_len_for_sample)
            input_ids_sample.append(self.sep_token_id)

            # Pad sequence to the fixed bert_input_seq_len
            padding_len = self.bert_input_seq_len - len(input_ids_sample)
            input_ids_sample.extend([self.pad_token_id] * padding_len)
            bert_input_ids_batch.append(input_ids_sample)

            # Store the corresponding size values, padded to bert_effective_max_len_for_size
            # Ensure we only take up to bert_len_for_sample values
            size_vals_sample = norm_size_values[i, :bert_len_for_sample].cpu().numpy()
            # Pad the extracted values if needed for consistent tensor shape later
            padded_size_vals = np.pad(
                size_vals_sample,
                (0, self.bert_effective_max_len_for_size - len(size_vals_sample)),
                mode='constant', constant_values=0
            )
            bert_size_values_batch.append(padded_size_vals)


        # Convert lists to tensors
        # Shape: [batch_size, bert_input_seq_len]
        input_ids_tensor = torch.tensor(bert_input_ids_batch, dtype=torch.long).to(device)
        # Shape: [batch_size, bert_effective_max_len_for_size]
        size_values_tensor = torch.tensor(np.array(bert_size_values_batch), dtype=torch.float32).to(device)
        # Shape: [batch_size, bert_input_seq_len]
        bert_attention_mask = (input_ids_tensor != self.pad_token_id).long()


        # --- Calculate BERT Embeddings with Continuous Value Injection ---
        # Get standard token embeddings
        # Shape: [batch_size, bert_input_seq_len, bert_hidden_size]
        word_embeddings = self.bert_model.base_model.embeddings.word_embeddings(input_ids_tensor)

        # Get standard position embeddings
        # Shape: [1, bert_input_seq_len] -> Expand
        position_ids = torch.arange(self.bert_input_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(input_ids_tensor)
        # Shape: [batch_size, bert_input_seq_len, bert_hidden_size]
        position_embeddings = self.bert_model.base_model.embeddings.position_embeddings(position_ids)

        # Initialize continuous embedding offsets
        # Shape: [batch_size, bert_input_seq_len, bert_hidden_size]
        continuous_embed_offset = torch.zeros_like(word_embeddings)

        # Add projected size values at the positions of [SIZE_VAL] tokens
        for j in range(self.bert_effective_max_len_for_size):
            # Index of the j-th [SIZE_VAL] token (0-indexed) is 1 + j
            size_token_seq_idx = 1 + j

            # Get the j-th size values for the whole batch
            # Shape: [batch_size, 1]
            current_size_vals = size_values_tensor[:, j].unsqueeze(-1)

            # Project values using the dedicated linear layer
            # Shape: [batch_size, bert_hidden_size]
            size_proj = self.bert_size_embed_proj(current_size_vals)

            # Create a mask for samples where the j-th token is actually part of the original sequence
            # Shape: [batch_size, 1]
            active_mask = (torch.tensor(bert_original_lengths_batch, device=device) > j).float().unsqueeze(-1)

            # Add the projection to the offset tensor at the correct position, applying the mask
            # Unsqueeze size_proj to allow broadcasting: [batch_size, 1, bert_hidden_size]
            continuous_embed_offset[:, size_token_seq_idx, :] += size_proj * active_mask


        # Combine base embeddings and continuous offsets
        # Shape: [batch_size, bert_input_seq_len, bert_hidden_size]
        inputs_embeds = word_embeddings + position_embeddings + continuous_embed_offset

        # Apply embedding LayerNorm and Dropout (mimicking BERT's standard embedding layer output)
        inputs_embeds = self.bert_model.base_model.embeddings.LayerNorm(inputs_embeds)
        inputs_embeds = self.bert_model.base_model.embeddings.dropout(inputs_embeds)

        # --- Pass through BERT Encoder ---
        # Use inputs_embeds and attention_mask
        bert_outputs = self.bert_model(
            inputs_embeds=inputs_embeds,
            attention_mask=bert_attention_mask,
            return_dict=True
        )

        # Use the output corresponding to the [CLS] token (index 0)
        # Shape: [batch_size, bert_hidden_size]
        bert_cls_output = bert_outputs.last_hidden_state[:, 0, :]
        # ======================== End BERT Path ==========================


        # ======================== Combine and Classify ========================
        # Concatenate the LSTM context vector and BERT [CLS] output
        # Shape: [batch_size, (lstm_hidden * dir) + bert_hidden]
        combined_features = torch.cat((lstm_context_vector, bert_cls_output), dim=1)

        # Pass through the final fully connected classification head
        # Output shape: [batch_size, 1] (raw logits)
        logits = self.fc_layers(combined_features)
        # ======================== End Combine and Classify ========================

        # Return raw logits; loss function (e.g., BCEWithLogitsLoss) will handle sigmoid
        return logits

