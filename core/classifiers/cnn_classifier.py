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

class CNNClassifier(BaseClassifier):
    """
        CNN model for binary classification of sequential data.
    """

    def __init__(self, normalization_params, kernel_width):
        """
            Creates an instance.
        """

        # Initialize
        super().__init__(normalization_params)
        self.class_name = self.__class__.__name__

        self.args = {
            'kernel_width': kernel_width,
        }
        
        # More layers for better reasoning
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, kernel_width), padding=(0, kernel_width // 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, kernel_width), padding=(0, kernel_width // 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, kernel_width), padding=(0, kernel_width // 2))
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate output size after convolutions (accounting for padding)
        self.pool = nn.AdaptiveAvgPool2d((1, self.max_len // 2))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * (self.max_len // 2), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
            Implements a forward pass.
        """

        # Input shape: [batch_size, 2, max_len]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, 2, max_len]
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        # Pooling
        x = self.pool(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
       
        # Return result
        return x