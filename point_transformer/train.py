"""
Training Script for Point Transformer

This script handles the complete training pipeline including data loading,
model initialization, training loop, and loss tracking.

Usage:
    python train.py --data_dir /path/to/data --n_points 10240 --batch_size 2 --epochs 100
"""

import os
import warnings
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import S3DISDataset
from model import PointTransformer

warnings.filterwarnings("ignore")


def normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize features using z-score normalization.
    
    Args:
        tensor (torch.Tensor): Input features to normalize
    
    Returns:
        torch.Tensor: Normalized features with mean=0, std=1
    """
    return (tensor - tensor.mean()) / (tensor.std() + 0.01)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Point Transformer model
        dataloader: Training data loader
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function (NLLLoss for log softmax output)
        device: Device to run training on (cuda or cpu)
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for coords, features, labels in dataloader:
        # Move data to device
        coords = coords.to(device)
        features = features.to(device)
        labels = labels.to(device)
        
        # Normalize coordinates and features
        features = normalize_features(features)
        coords = normalize_features(coords)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model([coords, features])
        
        # Compute loss (predictions are log probabilities)
        # Permute to (B, C, N) for NLLLoss
        loss = loss_fn(predictions.permute(0, 2, 1), labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    return epoch_loss / num_batches


def train(
    data_dir: str,
    n_points: int,
    num_classes: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int = 4,
    save_path: str = "checkpoints"
):
    """
    Complete training pipeline for Point Transformer.
    
    Args:
        data_dir: Path to directory containing training data
        n_points: Number of points to sample from each point cloud
        num_classes: Number of semantic classes
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        num_workers: Number of workers for data loading
        save_path: Directory to save model checkpoints
    """
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    print("Loading dataset...")
    train_dataset = S3DISDataset(data_dir, n_points)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    print(f"Dataset size: {len(train_dataset)} samples")
    
    # Initialize model
    print("Initializing model...")
    model = PointTransformer(
        input_dim=3,  # RGB features
        n_points=n_points,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.NLLLoss()
    
    # Training loop
    print("\nStarting training...")
    loss_history = []
    
    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        loss_history.append(avg_loss)
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0 and epoch > 0:
            checkpoint_path = os.path.join(
                save_path, 
                f"model_epoch_{epoch}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(save_path, "model_final.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history[-1],
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    loss_plot_path = os.path.join(save_path, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved: {loss_plot_path}")
    
    return model, loss_history


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train Point Transformer for semantic segmentation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to training data directory"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.95,
        help="Learning rate (default: 0.95)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Run training
    train(
        data_dir=args.data_dir,
        n_points=args.n_points,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()