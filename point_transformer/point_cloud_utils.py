"""
Point Cloud Utility Functions

This module provides essential operations for point cloud processing including
k-nearest neighbor search, farthest point sampling, and point indexing.

Functions:
    find_knn: Find k-nearest neighbors for each point
    index_points: Index points based on given indices
    farthest_point_sample: Sample points using farthest point sampling algorithm
"""

import time
import torch
import numpy as np
from scipy.spatial import distance
import fpsample


def find_knn(points: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    Find k-nearest neighbors for each point in the point cloud.
    
    This function computes pairwise distances between all points and returns
    the indices of the k nearest neighbors for each point.
    
    Args:
        points (torch.Tensor): Point coordinates of shape (B, N, 3)
            where B is batch size, N is number of points
        k (int): Number of nearest neighbors to find. Default: 16
    
    Returns:
        torch.Tensor: Indices of k-nearest neighbors of shape (B, N, k)
    
    Note:
        - The function excludes the point itself from its neighbors
        - Uses Euclidean distance for neighbor computation
        - Processes each batch element separately for memory efficiency
    
    Example:
        >>> points = torch.rand(2, 1024, 3)
        >>> neighbors = find_knn(points, k=16)
        >>> print(neighbors.shape)  # (2, 1024, 16)
    """
    points = points.to(torch.float32)
    
    # Initialize result tensor
    result = torch.zeros((*points.shape[:2], k)).to(points.device)
    
    # Process each batch separately
    for batch_idx in range(points.shape[0]):
        batch_points = points[batch_idx, :, :]
        
        # Compute pairwise distances using scipy
        # pdist computes condensed distance matrix, squareform converts to square
        pairwise_distances = distance.squareform(
            distance.pdist(batch_points.detach().cpu())
        )
        
        # Convert to tensor and move to device
        pairwise_distances = torch.tensor(pairwise_distances).to(points.device)
        
        # Sort distances and get k nearest neighbors
        # Start from index 1 to exclude the point itself (distance = 0)
        knn_indices = torch.argsort(pairwise_distances, dim=1)[:, 1:1+k]
        
        result[batch_idx, :, :] = knn_indices
    
    return result


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Index points based on provided indices to gather specific points.
    
    This function is used to gather neighbor points based on k-NN indices.
    
    Args:
        points (torch.Tensor): Point features of shape (B, N, C)
            where B is batch size, N is number of points, C is feature dimension
        indices (torch.Tensor): Indices to gather of shape (B, N, K)
            where K is number of neighbors
    
    Returns:
        torch.Tensor: Indexed points of shape (B, N, K, C)
    
    Example:
        >>> points = torch.rand(2, 1024, 3)
        >>> indices = torch.randint(0, 1024, (2, 1024, 16))
        >>> neighbors = index_points(points, indices)
        >>> print(neighbors.shape)  # (2, 1024, 16, 3)
    """
    # Expand points to match the k-neighbors dimension
    # (B, N, C) -> (B, N, 1, C) -> (B, N, K, C)
    expanded_points = points.unsqueeze(2).expand(-1, -1, indices.shape[2], -1)
    
    # Expand indices to match feature dimension
    # (B, N, K) -> (B, N, K, 1) -> (B, N, K, C)
    expanded_indices = indices.unsqueeze(3).expand(
        -1, -1, -1, points.shape[-1]
    ).to(torch.int64)
    
    # Gather the neighbor points based on indices
    neighbor_points = torch.gather(expanded_points, dim=1, index=expanded_indices)
    
    return neighbor_points


def farthest_point_sample(
    points: torch.Tensor, 
    n_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points using Farthest Point Sampling (FPS) algorithm.
    
    FPS iteratively selects points that are farthest from already selected points,
    ensuring good coverage of the point cloud geometry.
    
    Args:
        points (torch.Tensor): Input points of shape (B, N, 3) or (N, 3)
            where B is batch size, N is number of points
        n_samples (int): Number of points to sample
    
    Returns:
        tuple containing:
            - sampled_points (torch.Tensor): Sampled points of shape (B, n_samples, 3)
            - sampled_indices (torch.Tensor): Indices of sampled points (B, n_samples)
    
    Note:
        Uses the fpsample library for efficient FPS implementation
    
    Example:
        >>> points = torch.rand(2, 4096, 3)
        >>> sampled, indices = farthest_point_sample(points, 1024)
        >>> print(sampled.shape)  # (2, 1024, 3)
    """
    # Add batch dimension if not present
    if len(points.shape) < 3:
        points = points[None, ...]
    
    # Initialize output tensors
    sampled_points = torch.zeros(points.shape[0], n_samples, 3)
    sampled_indices = torch.zeros(points.shape[0], n_samples)
    
    # Process each batch separately
    for batch_idx in range(points.shape[0]):
        # Use fpsample library for efficient FPS
        fps_indices = fpsample.fps_sampling(points[batch_idx, :, :], n_samples)
        
        sampled_points[batch_idx, :, :] = points[batch_idx, fps_indices, :]
        sampled_indices[batch_idx, :] = torch.tensor(fps_indices)
    
    return sampled_points, sampled_indices