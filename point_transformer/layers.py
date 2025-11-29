"""
Point Transformer Neural Network Layers

This module implements the core layers of the Point Transformer architecture,
including the attention mechanism, blocks, and transition layers for encoding
and decoding point cloud features.

Classes:
    BatchNorm1d_P: BatchNorm for point cloud features (B, N, C)
    BatchNorm2d_P: BatchNorm for neighbor features (B, N, K, C)
    PointTransformerLayer: Self-attention layer for point clouds
    PointTransformerBlock: Residual block with Point Transformer layer
    TransitionDown: Downsampling layer using FPS and feature aggregation
    TransitionUp: Upsampling layer with feature interpolation
"""

import torch
from torch import nn
import torch.nn.functional as F

from point_cloud_utils import find_knn, index_points, farthest_point_sample


class BatchNorm1d_P(nn.BatchNorm1d):
    """
    Batch Normalization for point cloud features.
    
    Adapts standard BatchNorm1d to work with point cloud data of shape (B, N, C)
    by transposing before and after the normalization operation.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features of shape (B, N, C)
                where B=batch, N=points, C=channels
        
        Returns:
            torch.Tensor: Normalized features of shape (B, N, C)
        """
        # (B, N, C) -> (B, C, N) -> BatchNorm -> (B, N, C)
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class BatchNorm2d_P(nn.BatchNorm2d):
    """
    Batch Normalization for neighbor features.
    
    Adapts standard BatchNorm2d to work with neighbor data of shape (B, N, K, C)
    by permuting dimensions appropriately.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features of shape (B, N, K, C)
                where B=batch, N=points, K=neighbors, C=channels
        
        Returns:
            torch.Tensor: Normalized features of shape (B, N, K, C)
        """
        # (B, N, K, C) -> (B, C, N, K) -> BatchNorm -> (B, N, K, C)
        x = x.permute(0, 3, 1, 2)
        x = super().forward(x)
        return x.permute(0, 2, 3, 1)


class PointTransformerLayer(nn.Module):
    """
    Point Transformer Self-Attention Layer
    
    Implements the vector self-attention mechanism from the Point Transformer paper.
    Computes attention-weighted aggregation of neighbor features with positional encoding.
    
    The layer implements the equation:
    y_i = Σ_j ρ(γ(ψ(x_i) - φ(x_j) + δ)) ⊙ (α(x_j) + δ)
    
    where:
    - φ, ψ, α are linear transformations
    - γ is an MLP for attention
    - δ is positional encoding
    - ρ is softmax normalization
    
    Args:
        n_points (int): Number of points (unused, kept for compatibility)
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
    """
    
    def __init__(self, n_points: int, input_dim: int, output_dim: int):
        super().__init__()
        
        # Feature transformation for query
        self.phi = nn.Linear(output_dim, output_dim)
        
        # Feature transformation for key
        self.psi = nn.Linear(input_dim, output_dim)
        
        # Attention weight computation
        self.gamma = nn.Sequential(
            nn.Linear(output_dim, output_dim),  
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Positional encoding
        self.sigma = nn.Sequential(
            nn.Linear(3, 3),  
            nn.ReLU(),
            nn.Linear(3, output_dim)
        )
        
        # Value transformation
        self.alpha = nn.Linear(input_dim, output_dim)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self, 
        coords: torch.Tensor, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply point transformer attention.
        
        Args:
            coords (torch.Tensor): Point coordinates of shape (B, N, 3)
            features (torch.Tensor): Point features of shape (B, N, C)
        
        Returns:
            torch.Tensor: Transformed features of shape (B, N, output_dim)
        """
        # Add neighbor dimension to features: (B, N, C) -> (B, N, 1, C)
        features = features[:, :, None, :]
        
        # Find k-nearest neighbors
        knn_indices = find_knn(coords, k=16)
        knn_coords = index_points(coords, knn_indices)
        
        # Transform features
        xi = self.phi(features)  # Query: (B, N, 1, C)
        xj = self.psi(knn_coords)  # Key: (B, N, K, C)
        
        # Compute positional encoding
        pos_encoding = self.sigma(coords[:, :, None, :] - knn_coords)
        
        # Compute attention weights
        attention = self.gamma(xi - xj + pos_encoding)
        attention = self.softmax(attention)
        
        # Compute values with positional encoding
        values = self.alpha(knn_coords) + pos_encoding
        
        # Apply attention and aggregate over neighbors
        output = torch.sum(attention * values, dim=2)
        
        return output


class PointTransformerBlock(nn.Module):
    """
    Point Transformer Residual Block
    
    Implements a residual block with Point Transformer attention layer.
    Structure: Linear -> PT Layer -> Linear + Residual
    
    Args:
        n_points (int): Number of points
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
        share_planes (int): Shared planes (unused, kept for compatibility)
        nsample (int): Number of samples (unused, kept for compatibility)
    """
    
    def __init__(
        self, 
        n_points: int, 
        input_dim: int, 
        output_dim: int, 
        share_planes: int = 8, 
        nsample: int = 16
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(output_dim, output_dim, bias=False)
        self.bn1 = BatchNorm1d_P(output_dim)

        self.transformer = PointTransformerLayer(n_points, input_dim, output_dim)
        self.bn2 = BatchNorm1d_P(output_dim)

        self.linear3 = nn.Linear(output_dim, output_dim, bias=False)
        self.bn3 = BatchNorm1d_P(output_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, px: list) -> list:
        """
        Apply Point Transformer block.
        
        Args:
            px (list): [coords, features]
                - coords (torch.Tensor): Point coordinates (B, N, 3)
                - features (torch.Tensor): Point features (B, N, C)
        
        Returns:
            list: [coords, new_features]
                - coords (torch.Tensor): Unchanged point coordinates (B, N, 3)
                - new_features (torch.Tensor): Transformed features (B, N, C)
        """
        coords, features = px
        identity = features
        
        # First linear layer
        features = self.relu(self.linear1(features))
        
        # Point Transformer attention
        features = self.relu(self.transformer(coords, features))
        
        # Final linear layer
        features = self.linear3(features)
        
        # Residual connection
        features += identity
        features = self.relu(features)
        
        return [coords, features]


class TransitionDown(nn.Module):
    """
    Downsampling Transition Layer
    
    Reduces the number of points using Farthest Point Sampling (FPS) and
    aggregates features from k-nearest neighbors using an MLP and max pooling.
    
    Args:
        n_points (int): Target number of points after downsampling
        k_neighbors (int): Number of neighbors to aggregate over
        input_dim (int): Dimension of input features
    """
    
    def __init__(self, n_points: int, k_neighbors: int, input_dim: int):
        super().__init__()
        self.n_points = n_points
        self.k_neighbors = k_neighbors
        
        # MLP for feature aggregation followed by max pooling over neighbors
        self.knn_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), 
            nn.ReLU(), 
            nn.MaxPool2d((k_neighbors, 1))
        )
    
    def forward(self, px: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample points and aggregate features.
        
        Args:
            px (list): [coords, features]
                - coords (torch.Tensor): Point coordinates (B, N, 3)
                - features (torch.Tensor): Point features (B, N, C)
        
        Returns:
            tuple:
                - new_coords (torch.Tensor): Sampled coordinates (B, n_points, 3)
                - new_features (torch.Tensor): Aggregated features (B, n_points, C*2)
        """
        coords, features = px
        
        # Farthest point sampling to select representative points
        new_coords, new_coords_idx = farthest_point_sample(
            coords.detach(), 
            self.n_points
        )
        
        # Find k-nearest neighbors for each sampled point
        knn_indices = find_knn(new_coords, self.k_neighbors)
        knn_features = index_points(features, knn_indices)
        
        # Aggregate features using MLP and max pooling
        aggregated_features = self.knn_mlp(knn_features)
        
        return new_coords, aggregated_features.squeeze()


class TransitionUp(nn.Module):
    """
    Upsampling Transition Layer
    
    Increases the number of points through interpolation and combines features
    from two resolutions using skip connections.
    
    Args:
        input_dim (int): Dimension of input features from lower resolution
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Linear layers to transform features before concatenation
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.linear2 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
    
    def forward(self, px1_px2: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Upsample features and combine with skip connection.
        
        Args:
            px1_px2 (list): [coords1, features1, coords2, features2]
                - coords1: Lower resolution coordinates (B, N1, 3)
                - features1: Lower resolution features (B, N1, C)
                - coords2: Higher resolution coordinates (B, N2, 3)
                - features2: Higher resolution features (B, N2, C//2)
        
        Returns:
            tuple:
                - coords2: Higher resolution coordinates (B, N2, 3)
                - combined_features: Upsampled and combined features (B, N2, C//2)
        """
        coords1, features1, coords2, features2 = px1_px2
        
        # Transform lower resolution features
        features1 = self.linear1(features1)
        
        # Interpolate to match higher resolution
        features1 = F.interpolate(
            features1.permute(0, 2, 1),  # (B, C, N1)
            size=features2.shape[1],      # Interpolate to N2 points
            mode='linear'
        ).permute(0, 2, 1)  # Back to (B, N2, C)
        
        # Combine with skip connection
        return coords2, features1 + features2