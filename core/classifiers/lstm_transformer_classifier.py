import math
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
from .base_classifier import BaseClassifier

class LSTMTransformerClassifier(BaseClassifier):
    """
    Bidirectional LSTM
    Multi-head attention mechanism (with configurable number of heads)
    Positional encoding (similar to transformer architectures)
    Residual connections and layer normalization
    Feed-forward network component after attention
    Sigmoid output for binary classification
    """

    def __init__(self, normalization_params, 
                 # Parameterized configuration
                 hidden_size=128, 
                 num_layers=2, 
                 dropout_rate=0.3,
                 embedding_dim=32,
                 fc_dims=[128, 64],
                 bidirectional=True,
                 num_heads=8,
                 attention_dropout=0.1):
        """
        Creates an instance with parameterized configuration.
        
        Args:
            kernel_width: Width of the kernel
            max_len: Maximum sequence length
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            embedding_dim: Dimension of embedding layers
            fc_dims: List of dimensions for fully connected layers
            bidirectional: Whether to use bidirectional LSTM
            num_heads: Number of attention heads
            attention_dropout: Dropout rate for attention layers
        """
        # Initialize
        super().__init__(normalization_params)
        self.class_name = self.__class__.__name__

        # Saves members
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.fc_dims = fc_dims
        self.num_heads = num_heads
        self.args = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'embedding_dim': embedding_dim,
            'fc_dims': fc_dims,
            'bidirectional': bidirectional,
            'num_heads': num_heads,
            'attention_dropout': attention_dropout
        }
        
        # Feature embedding layers
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        self.size_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
        # Combined feature dimension after embedding
        self.feature_dim = embedding_dim * 2
        
        # Direction factor (1 for unidirectional, 2 for bidirectional)
        self.direction_factor = 2 if bidirectional else 1
        
        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Define the output dimension of LSTM
        lstm_output_dim = hidden_size * self.direction_factor
        
        # Ensure the hidden size is divisible by the number of heads
        assert lstm_output_dim % num_heads == 0, "Hidden size must be divisible by the number of attention heads"
        
        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Layer normalization for attention stability
        self.layer_norm1 = nn.LayerNorm(lstm_output_dim)
        self.layer_norm2 = nn.LayerNorm(lstm_output_dim)
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim * 4, lstm_output_dim)
        )
        
        # Dynamically build fully connected layers for classification
        fc_layers = []
        input_dim = lstm_output_dim
        
        for dim in fc_dims:
            fc_layers.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = dim
            
        # Final output layer
        fc_layers.append(nn.Linear(input_dim, 1))
        
        # Combine all FC layers
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Position embedding for enhancing positional information
        self.pos_encoder = PositionalEncoding(lstm_output_dim, dropout_rate, self.max_len)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights for faster convergence.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def _create_attention_mask(self, mask):
        """
        Creates attention mask for multihead attention.
        The mask is inverted because MultiheadAttention uses True values to indicate positions to ignore.
        """
        # Create a mask for padding (False for valid tokens, True for padding)
        attn_mask = ~mask.bool()  # Invert mask for nn.MultiheadAttention
        return attn_mask
        
    def forward(self, x):
        """
        Implements a forward pass.
        """
        batch_size = x.size(0)
        
        # Create mask for padding (1 for valid tokens, 0 for padding)
        mask = (x.sum(dim=1) != 0).float()
        seq_lengths = mask.sum(dim=1).long()
        
        # Split time and size features
        time_features = x[:, 0, :].unsqueeze(2)
        size_features = x[:, 1, :].unsqueeze(2)
        
        # Apply embedding layers
        time_embedded = self.time_embedding(time_features)
        size_embedded = self.size_embedding(size_features)
        
        # Concatenate embedded features
        embedded = torch.cat([time_embedded, size_embedded], dim=2)
        
        # Pack the padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            lengths=seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_output, _ = self.lstm(packed_embedded)
        
        # Unpack the sequence
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=True,
            total_length=self.max_len
        )
        
        # Add positional encoding
        lstm_output = self.pos_encoder(lstm_output)
        
        # Create attention mask from padding mask
        attn_mask = self._create_attention_mask(mask)
        
        # Apply multi-head self-attention with residual connection and layer normalization
        attn_output, _ = self.multihead_attn(
            query=lstm_output, 
            key=lstm_output, 
            value=lstm_output,
            key_padding_mask=attn_mask
        )
        
        # Residual connection and layer normalization
        attn_output = self.layer_norm1(lstm_output + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm2(attn_output + ffn_output)
        
        # Average pooling across sequence length (ignoring padding)
        masked_output = output * mask.unsqueeze(2)
        seq_sum = masked_output.sum(dim=1)
        seq_lengths_expanded = seq_lengths.unsqueeze(1).expand(-1, output.size(2))
        pooled_output = seq_sum / seq_lengths_expanded
        
        # Pass through fully connected layers
        output = self.fc_layers(pooled_output)
        
        # Apply sigmoid for binary classification
        output = torch.sigmoid(output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequential data.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)