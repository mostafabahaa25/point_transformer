"""
S3DIS Dataset Module

This module handles loading and preprocessing of the S3DIS (Stanford Large-Scale 3D Indoor Spaces) 
dataset for point cloud semantic segmentation tasks.

Classes:
    S3DISDataset: Custom PyTorch Dataset for loading S3DIS point cloud data
"""

import os
import torch
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    """
    S3DIS Point Cloud Dataset
    
    This dataset loads point cloud data from the S3DIS dataset, which contains
    3D scans of indoor spaces with semantic segmentation labels.
    
    Attributes:
        n_points (int): Number of points to sample from each point cloud
        root_dir (str): Root directory containing the dataset files
        data_paths (list): List of file paths to individual point cloud files
    
    Args:
        root_dir (str): Path to the directory containing .pth files
        n_points (int): Number of points to randomly sample from each cloud
    
    Example:
        >>> dataset = S3DISDataset('/path/to/s3dis/Area_4', n_points=10240)
        >>> coords, features, labels = dataset[0]
        >>> print(coords.shape)  # (10240, 3)
    """
    
    def __init__(self, root_dir: str, n_points: int):
        """
        Initialize the S3DIS dataset.
        
        Args:
            root_dir: Directory containing .pth point cloud files
            n_points: Number of points to sample per point cloud
        """
        self.n_points = n_points
        self.root_dir = root_dir
        self.data_paths = []
        
        # Collect all .pth files in the directory
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            self.data_paths.append(file_path)
    
    def __len__(self) -> int:
        """Return the total number of point clouds in the dataset."""
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and return a single point cloud sample.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            tuple containing:
                - coords (torch.Tensor): Point coordinates of shape (n_points, 3)
                - features (torch.Tensor): RGB color features of shape (n_points, 3)
                - labels (torch.Tensor): Semantic labels of shape (n_points,)
        
        Note:
            Points are randomly sampled to match the specified n_points.
            All tensors are converted to appropriate dtypes (float32 for coords/features,
            long for labels).
        """
        # Load point cloud data from file
        file = torch.load(self.data_paths[idx])
        coords = file["coord"]
        features = file["color"]
        labels = file["semantic_gt"]
        
        # Randomly sample points to get fixed number of points
        sample_indices = torch.randint(
            low=0, 
            high=coords.shape[0], 
            size=(self.n_points,)
        )

        # Extract sampled points and convert to appropriate types
        coords = torch.tensor(coords[sample_indices, :]).to(torch.float32)
        features = torch.tensor(features[sample_indices, :]).to(torch.float32)
        labels = torch.tensor(labels[sample_indices]).squeeze().long()
        
        return coords, features, labels