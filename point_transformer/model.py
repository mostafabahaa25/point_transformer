"""
Point Transformer Model

This module implements the complete Point Transformer architecture for semantic
segmentation of 3D point clouds. The model uses a U-Net-like encoder-decoder
structure with Point Transformer attention blocks.

Classes:
    PointTransformer: Complete Point Transformer network for semantic segmentation
"""

import torch
from torch import nn

from layers import (
    PointTransformerBlock,
    TransitionDown,
    TransitionUp
)


class PointTransformer(nn.Module):
    """
    Point Transformer Network for Semantic Segmentation
    
    Implements a hierarchical encoder-decoder architecture with Point Transformer
    attention mechanisms. The network progressively downsamples the point cloud,
    processes it at multiple scales, then upsamples back to the original resolution.
    
    Architecture:
        - Encoder: 5 levels of downsampling (N → N/4 → N/16 → N/64 → N/256)
        - Each level has Point Transformer blocks for feature processing
        - Decoder: 4 levels of upsampling with skip connections
        - Output: Per-point semantic class predictions
    
    Args:
        input_dim (int): Dimension of input features (typically 3 for RGB)
        n_points (int): Number of input points (must be divisible by 256)
        num_classes (int): Number of semantic classes to predict
    
    Example:
        >>> model = PointTransformer(input_dim=3, n_points=4096, num_classes=13)
        >>> coords = torch.rand(2, 4096, 3)
        >>> features = torch.rand(2, 4096, 3)
        >>> predictions = model([coords, features])
        >>> print(predictions.shape)  # (2, 4096, 13)
    """
    
    def __init__(self, input_dim: int, n_points: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.n_points = n_points
        self.num_classes = num_classes
        
        # Initial feature extraction
        self.mlp0 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 32)
        )
        
        # Encoder Level 0: Full resolution (N points, 32 channels)
        self.pt0 = PointTransformerBlock(n_points, 3, 32)
        
        # Encoder Level 1: N/4 points, 64 channels
        self.td1 = TransitionDown(n_points // 4, 16, 32)
        self.pt1 = PointTransformerBlock(n_points // 4, 3, 64)
        
        # Encoder Level 2: N/16 points, 128 channels
        self.td2 = TransitionDown(n_points // 16, 16, 64)
        self.pt2 = PointTransformerBlock(n_points // 16, 3, 128)
        
        # Encoder Level 3: N/64 points, 256 channels
        self.td3 = TransitionDown(n_points // 64, 16, 128)
        self.pt3 = PointTransformerBlock(n_points // 64, 3, 256)
        
        # Encoder Level 4 (Bottleneck): N/256 points, 512 channels
        self.td4 = TransitionDown(n_points // 256, 16, 256)
        self.pt4 = PointTransformerBlock(n_points // 256, 3, 512)
        
        # Bottleneck processing
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.pt5 = PointTransformerBlock(n_points // 256, 3, 512)
        
        # Decoder Level 1: Upsample from N/256 to N/64
        self.tu6 = TransitionUp(512)
        self.pt6 = PointTransformerBlock(n_points // 64, 3, 256)
        
        # Decoder Level 2: Upsample from N/64 to N/16
        self.tu7 = TransitionUp(256)
        self.pt7 = PointTransformerBlock(n_points // 16, 3, 128)
        
        # Decoder Level 3: Upsample from N/16 to N/4
        self.tu8 = TransitionUp(128)
        self.pt8 = PointTransformerBlock(n_points // 4, 3, 64)
        
        # Decoder Level 4: Upsample from N/4 to N (full resolution)
        self.tu9 = TransitionUp(64)
        self.pt9 = PointTransformerBlock(n_points // 4, 3, 32)
        
        # Final classification head
        self.mlp3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, px: list) -> torch.Tensor:
        """
        Forward pass through the Point Transformer network.
        
        Args:
            px (list): [coords, features]
                - coords (torch.Tensor): Point coordinates of shape (B, N, 3)
                - features (torch.Tensor): Point features of shape (B, N, C)
        
        Returns:
            torch.Tensor: Log probabilities for each class, shape (B, N, num_classes)
        
        Note:
            The output uses log_softmax, suitable for use with NLLLoss.
            For actual probabilities, apply torch.exp() to the output.
        """
        coords, features = px
        
        # ===== ENCODER =====
        # Level 0: Initial feature extraction
        features0 = self.mlp0(features)
        coords0, features0 = self.pt0([coords, features0])
        
        # Level 1: Downsample to N/4 points
        coords1, features1 = self.td1([coords0, features0])
        coords1, features1 = self.pt1([coords1, features1])
        
        # Level 2: Downsample to N/16 points
        coords2, features2 = self.td2([coords1, features1])
        coords2, features2 = self.pt2([coords2, features2])
        
        # Level 3: Downsample to N/64 points
        coords3, features3 = self.td3([coords2, features2])
        coords3, features3 = self.pt3([coords3, features3])
        
        # Level 4 (Bottleneck): Downsample to N/256 points
        coords4, features4 = self.td4([coords3, features3])
        coords4, features4 = self.pt4([coords4, features4])
        
        # ===== BOTTLENECK =====
        features5 = self.mlp2(features4)
        coords6, features6 = self.pt5([coords4, features5])
        
        # ===== DECODER =====
        # Level 1: Upsample and combine with encoder level 3
        coords7, features7 = self.tu6([coords6, features6, coords3, features3])
        coords7, features7 = self.pt6([coords7, features7])
        
        # Level 2: Upsample and combine with encoder level 2
        coords8, features8 = self.tu7([coords7, features7, coords2, features2])
        coords8, features8 = self.pt7([coords8, features8])
        
        # Level 3: Upsample and combine with encoder level 1
        coords9, features9 = self.tu8([coords8, features8, coords1, features1])
        coords9, features9 = self.pt8([coords9, features9])
        
        # Level 4: Upsample and combine with encoder level 0
        coords10, features10 = self.tu9([coords9, features9, coords0, features0])
        coords10, features10 = self.pt9([coords10, features10])
        
        # ===== OUTPUT =====
        # Final classification
        logits = self.mlp3(features10)
        output = self.log_softmax(logits)
        
        return output