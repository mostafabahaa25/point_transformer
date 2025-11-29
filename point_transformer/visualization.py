"""
Visualization Script for Point Transformer Results

This script loads a trained model and visualizes the semantic segmentation
predictions on point clouds using Open3D.

Usage:
    python visualize.py --checkpoint checkpoints/model_final.pth --data_path /path/to/scene.pth
"""

import argparse
import random
import numpy as np
import torch
import open3d as o3d

from dataset import S3DISDataset
from model import PointTransformer


def generate_colors(n_classes: int) -> np.ndarray:
    """
    Generate random RGB colors for each semantic class.
    
    Args:
        n_classes (int): Number of semantic classes
    
    Returns:
        np.ndarray: Array of RGB colors, shape (n_classes, 3)
    """
    # Generate random hex colors
    hex_colors = [
        "#%06x" % random.randint(0, 0xFFFFFF) 
        for _ in range(n_classes)
    ]
    
    # Convert hex to RGB values (0-255)
    rgb_colors = np.array([
        tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        for h in hex_colors
    ])
    
    return rgb_colors


def normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize features using z-score normalization.
    
    Args:
        tensor (torch.Tensor): Input features to normalize
    
    Returns:
        torch.Tensor: Normalized features with mean=0, std=1
    """
    return (tensor - tensor.mean()) / (tensor.std() + 0.01)


def visualize_predictions(
    model_path: str,
    data_path: str,
    n_points: int,
    num_classes: int,
    device: torch.device
):
    """
    Load a trained model and visualize predictions on a point cloud.
    
    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to point cloud data file (.pth)
        n_points: Number of points to sample
        num_classes: Number of semantic classes
        device: Device to run inference on
    """
    # Initialize model
    print("Loading model...")
    model = PointTransformer(
        input_dim=3,
        n_points=n_points,
        num_classes=num_classes
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Load data
    print("Loading point cloud...")
    data = torch.load(data_path)
    coords = data["coord"]
    features = data["color"]
    labels = data["semantic_gt"]
    
    # Sample points
    torch.manual_seed(0)
    sample_indices = torch.randint(
        low=0, 
        high=coords.shape[0], 
        size=(n_points,)
    )
    
    coords = torch.tensor(coords[sample_indices, :]).unsqueeze(0).float()
    features = torch.tensor(features[sample_indices, :]).unsqueeze(0).float()
    labels = torch.tensor(labels[sample_indices]).unsqueeze(0)
    
    # Move to device and normalize
    coords = coords.to(device)
    features = features.to(device)
    
    features = normalize_features(features)
    coords_normalized = normalize_features(coords)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model([coords_normalized, features])
        predicted_classes = torch.argmax(predictions, dim=-1)
    
    # Generate colors for visualization
    class_colors = generate_colors(num_classes)
    
    # Create point cloud with predicted colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        coords[0, :].cpu().numpy()
    )
    pcd.colors = o3d.utility.Vector3dVector(
        class_colors[predicted_classes[0, :].cpu().numpy()] / 255.0
    )
    
    # Visualize
    print("Launching visualization...")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Q or ESC: Close window")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Transformer Semantic Segmentation",
        width=1280,
        height=720
    )


def visualize_from_dataloader(
    model_path: str,
    data_dir: str,
    n_points: int,
    num_classes: int,
    device: torch.device,
    batch_idx: int = 0
):
    """
    Load a trained model and visualize predictions from a dataset.
    
    Args:
        model_path: Path to saved model checkpoint
        data_dir: Directory containing dataset
        n_points: Number of points to sample
        num_classes: Number of semantic classes
        device: Device to run inference on
        batch_idx: Index of sample to visualize (default: 0)
    """
    # Initialize model
    print("Loading model...")
    model = PointTransformer(
        input_dim=3,
        n_points=n_points,
        num_classes=num_classes
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = S3DISDataset(data_dir, n_points)
    coords, features, labels = dataset[batch_idx]
    
    # Add batch dimension and move to device
    coords = coords.unsqueeze(0).to(device)
    features = features.unsqueeze(0).to(device)
    
    # Normalize
    features = normalize_features(features)
    coords_normalized = normalize_features(coords)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model([coords_normalized, features])
        predicted_classes = torch.argmax(predictions, dim=-1)
    
    # Generate colors for visualization
    class_colors = generate_colors(num_classes)
    
    # Create point cloud with predicted colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        coords[0, :].cpu().numpy()
    )
    pcd.colors = o3d.utility.Vector3dVector(
        class_colors[predicted_classes[0, :].cpu().numpy()] / 255.0
    )
    
    # Visualize
    print("Launching visualization...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Transformer Semantic Segmentation",
        width=1280,
        height=720
    )


def main():
    """Parse arguments and run visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize Point Transformer predictions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to single point cloud file (.pth)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to visualize (when using data_dir)"
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=10240,
        help="Number of points to sample (default: 10240)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=50,
        help="Number of semantic classes (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run visualization
    if args.data_path is not None:
        visualize_predictions(
            model_path=args.checkpoint,
            data_path=args.data_path,
            n_points=args.n_points,
            num_classes=args.num_classes,
            device=device
        )
    elif args.data_dir is not None:
        visualize_from_dataloader(
            model_path=args.checkpoint,
            data_dir=args.data_dir,
            n_points=args.n_points,
            num_classes=args.num_classes,
            device=device,
            batch_idx=args.sample_idx
        )
    else:
        print("Error: Must provide either --data_path or --data_dir")
        return


if __name__ == "__main__":
    main()