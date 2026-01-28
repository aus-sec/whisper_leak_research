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

class AttentionBiLSTMClassifier(BaseClassifier):
    """
    Bidirectional LSTM Neural Network with Attention Mechanism
    """

    def __init__(self, norm, 
                 # Parameterized configuration
                 hidden_size=128, 
                 num_layers=2,
                 dropout_rate=0.3,
                 embedding_dim=32,
                 fc_dims=[128, 64],
                 bidirectional=True,
                 attention_dim=128):
        """
        Creates an instance with parameterized configuration.
        
        Args:
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            embedding_dim: Dimension of embedding layers
            fc_dims: List of dimensions for fully connected layers
            bidirectional: Whether to use bidirectional LSTM
            attention_dim: Dimension of the attention layer
        """

        # Initialize
        super().__init__(norm)
        self.class_name = self.__class__.__name__

        # Saves members
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.fc_dims = fc_dims
        self.attention_dim = attention_dim

        self.args = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'embedding_dim': embedding_dim,
            'fc_dims': fc_dims,
            'bidirectional': bidirectional,
            'attention_dim': attention_dim
        }
        
        # Feature embedding layers
        inputs = 2 # time z, size z
        self.input_embeddings = []
        for i in range(inputs):
            self.input_embeddings.append(
                nn.Sequential(
                    nn.Linear(1, embedding_dim),
                    nn.ReLU(),
                    nn.LayerNorm(embedding_dim)
                )
            )
        self.input_embeddings = nn.ModuleList(self.input_embeddings)
        
        # Combined feature dimension after embedding
        self.feature_dim = embedding_dim * inputs
        
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
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.direction_factor, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Dynamically build fully connected layers
        fc_layers = []
        input_dim = hidden_size * self.direction_factor
        
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
    
    def _apply_attention(self, lstm_output, mask=None):
        """
        Apply attention mechanism to LSTM output.
        """
        attention_weights = self.attention(lstm_output)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(2) == 0, float('-inf'))
        
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights
        
    def forward(self, x):
        """
        Implements a forward pass.
        """
        batch_size = x.size(0)
        
        # Create mask for padding
        mask = (x.sum(dim=1) != 0).float()
        
        # Apply embeddings to each feature
        result = []
        for i, embedding in enumerate(self.input_embeddings):
            result.append(embedding(x[:, i, :].unsqueeze(2)))

        # Concatenate embedded features
        embedded = torch.cat(result, dim=2)
        
        # Pack the padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            lengths=mask.sum(dim=1).cpu().int(),
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
        
        # Apply attention mechanism
        context_vector, _ = self._apply_attention(lstm_output, mask)
        
        # Pass through fully connected layers
        output = self.fc_layers(context_vector)
        
        return output