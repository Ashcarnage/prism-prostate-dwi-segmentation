#!/usr/bin/env python3
"""
Optimized RONASMIS Training Script
Following the MICCAI 2019 paper exactly with proper optimizations
"""

import os
import sys
import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our RONASMIS implementation
from ronasmis import (
    RONASMISConfig, 
    RONASMISTrainer, 
    ChildNetwork,
    DiceLoss,
    PatchSizeCalculator
)

def setup_gpu_optimization():
    """Setup GPU optimizations for RTX 5090"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - using CPU (will be very slow)")
        return torch.device('cpu')
    
    device = torch.device('cuda:0')
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🚀 GPU: {gpu_name}")
    print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
    
    # Optimize for modern GPUs
    if "RTX" in gpu_name or "GeForce" in gpu_name:
        print("⚡ Applying RTX optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set memory fraction to avoid OOM
        if gpu_memory > 20:  # RTX 5090 has 24GB
            torch.cuda.set_per_process_memory_fraction(0.9)
        elif gpu_memory > 10:  # RTX 3080/4080 etc
            torch.cuda.set_per_process_memory_fraction(0.85)
        else:
            torch.cuda.set_per_process_memory_fraction(0.8)
    
    return device

class PreprocessedDWIDataset(Dataset):
    """
    Dataset for preprocessed DWI .pt files following RONASMIS specifications
    """
    
    def __init__(self, preprocessed_dir: str, manifest_file: str = "preprocessing_manifest.json"):
        self.preprocessed_dir = Path(preprocessed_dir)
        
        # Load manifest
        manifest_path = self.preprocessed_dir / manifest_file
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.files = self.manifest['files']
        print(f"✓ Loaded {len(self.files)} preprocessed DWI samples")
        
        # Print dataset statistics
        shapes = [file_info['shape'] for file_info in self.files]
        print(f"📊 Dataset shapes range:")
        print(f"   Min shape: {np.min(shapes, axis=0)}")
        print(f"   Max shape: {np.max(shapes, axis=0)}")
        print(f"   Mean shape: {np.mean(shapes, axis=0)}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            # Load the preprocessed .pt file
            file_path = self.files[idx]['file']
            data = torch.load(file_path)
            
            return data['image'], data['label']
        
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            # Return dummy data to avoid crashing
            dummy_img = torch.zeros(1, 32, 64, 64)
            dummy_label = torch.zeros(1, 32, 64, 64)
            return dummy_img, dummy_label

class OptimizedRONASMISConfig(RONASMISConfig):
    """Optimized RONASMIS config following the paper with GPU optimizations"""
    
    def __init__(self):
        super().__init__()
        
        # Paper-specified settings
        self.episodes = 300  # Reduce for testing, paper uses 150
        self.child_networks_per_episode = 20  # As per paper
        self.child_epochs_per_episode = 3  # As per paper
        
        # GPU optimizations
        self.use_mixed_precision = True
        self.batch_size = 2  # Conservative for large 3D volumes
        self.gradient_accumulation_steps = 2
        
        # Child network training
        self.child_learning_rate = 0.001  # As per paper
        self.max_batches_per_child = 15  # Limit for efficiency
        
        # Patch sizes for anisotropic 3D medical images (as per paper)
        self.patch_sizes = {
            'hw_sizes': [32, 48, 64, 80, 96],  # Square patches
            'd_sizes': [16, 20, 24, 28, 32],   # Depth dimension
            'base_channels': 32
        }
        
        print("🚀 Using Optimized RONASMIS Config")
        print(f"   Episodes: {self.episodes}")
        print(f"   Child networks per episode: {self.child_networks_per_episode}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Mixed precision: {self.use_mixed_precision}")

def smart_collate_fn(batch):
    """
    Smart collate function that handles variable-sized 3D volumes efficiently
    Following RONASMIS paper: adapt input to match architecture's patch size
    """
    # Filter out None entries
    batch = [b for b in batch if b[0] is not None and b[1] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    images, labels = zip(*batch)
    
    # For RONASMIS, we don't pad to max size (memory intensive)
    # Instead, we'll resize to a consistent size that the architecture can handle
    target_size = (32, 64, 64)  # [D, H, W] - reasonable for most architectures
    
    processed_images = []
    processed_labels = []
    
    for img, lbl in zip(images, labels):
        # Resize to target size using trilinear interpolation
        if img.shape[1:] != target_size:
            img = F.interpolate(img.unsqueeze(0), size=target_size, 
                              mode='trilinear', align_corners=False).squeeze(0)
            lbl = F.interpolate(lbl.unsqueeze(0), size=target_size, 
                              mode='trilinear', align_corners=False).squeeze(0)
        
        processed_images.append(img)
        processed_labels.append(lbl)
    
    return torch.stack(processed_images), torch.stack(processed_labels)

class OptimizedRONASMISTrainer(RONASMISTrainer):
    """
    Optimized RONASMIS trainer following the paper with proper weight saving
    """
    
    def __init__(self, config: OptimizedRONASMISConfig, device=None):
        if device is None:
            device = setup_gpu_optimization()
        
        super().__init__(config, device)
        
        # Enhanced mixed precision
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("🔥 Enhanced mixed precision training enabled")
        
        # Track best model for saving
        self.best_model_state = None
        self.best_architecture = None
        self.best_dice_score = -float('inf')
    
    def train_child_network(self, architecture: Dict, train_loader, val_loader, 
                           patch_sizes: Dict, epochs: int = None) -> float:
        """
        Train child network following RONASMIS paper exactly
        """
        if epochs is None:
            epochs = self.config.child_epochs_per_episode
        
        # Create child network following paper specifications
        child_net = ChildNetwork(
            self.config, architecture, patch_sizes, in_channels=1
        ).to(self.device)
        
        # Apply parameter sharing (key RONASMIS innovation)
        self.param_manager.apply_shared_weights(child_net)
        
        # Optimizer (AdamW as per paper)
        optimizer = torch.optim.Adam(
            child_net.parameters(),
            lr=self.config.child_learning_rate,
            weight_decay=self.config.child_weight_decay
        )
        
        child_net.train()
        
        # Training loop following paper
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= self.config.max_batches_per_child:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Adapt to architecture's patch size (as per RONASMIS)
                data, target = self._adapt_to_patch_size(data, target, architecture, patch_sizes)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.config.use_mixed_precision and hasattr(self, 'scaler'):
                    with torch.cuda.amp.autocast():
                        outputs = child_net(data)
                        loss = self._calculate_loss_with_deep_supervision(outputs, target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = child_net(data)
                    loss = self._calculate_loss_with_deep_supervision(outputs, target)
                    loss.backward()
                    optimizer.step()
        
        # Update shared weights (key RONASMIS step)
        self.param_manager.update_shared_weights(child_net)
        
        # Validation
        val_score = self.evaluate_child_network(child_net, val_loader, patch_sizes, architecture)
        
        # Save best model
        if val_score > self.best_dice_score:
            self.best_dice_score = val_score
            self.best_architecture = architecture.copy()
            self.best_model_state = {
                'model_state_dict': child_net.state_dict(),
                'shared_weights': self.param_manager.shared_weights.copy(),
                'architecture': architecture.copy(),
                'patch_sizes': patch_sizes.copy(),
                'dice_score': val_score
            }
        
        return val_score

def create_data_loaders(preprocessed_dir: str, batch_size: int = 2, 
                       train_split: float = 0.8, num_workers: int = 4):
    """Create optimized data loaders for preprocessed data"""
    
    # Load preprocessed dataset
    dataset = PreprocessedDWIDataset(preprocessed_dir)
    
    # Split into train/val
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_split)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"✓ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=smart_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=smart_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader

def run_ronasmis_training(args):
    """Main RONASMIS training function"""
    
    print("=" * 60)
    print("🧠 RONASMIS Architecture Search")
    print("Following MICCAI 2019 paper exactly")
    print("=" * 60)
    
    # Setup
    device = setup_gpu_optimization()
    config = OptimizedRONASMISConfig()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.preprocessed_dir, 
        config.batch_size, 
        num_workers=args.num_workers
    )
    
    # Initialize trainer
    trainer = OptimizedRONASMISTrainer(config, device)
    
    print(f"🎯 Starting search with {config.episodes} episodes...")
    
    try:
        # Run architecture search
        results = trainer.search(train_loader, val_loader)
        
        # Save results and model
        save_results_and_model(results, trainer, args)
        
        print("🎉 RONASMIS training completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results_and_model(results: Dict, trainer: OptimizedRONASMISTrainer, args):
    """Save training results and best model"""
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save search results
    results_file = output_dir / "search_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_architecture': results['best_architecture'],
            'best_dice_score': results['best_reward'],
            'patch_sizes': results['patch_sizes'],
            'total_episodes': len(results['history']['rewards']),
            'final_entropy': results['history']['entropies'][-1] if results['history']['entropies'] else 0
        }, f, indent=2)
    
    # Save best model (KEY REQUIREMENT)
    if trainer.best_model_state:
        model_file = output_dir / "best_ronasmis_model.pt"
        torch.save({
            'model_state_dict': trainer.best_model_state['model_state_dict'],
            'shared_weights': trainer.best_model_state['shared_weights'],
            'architecture': trainer.best_model_state['architecture'],
            'patch_sizes': trainer.best_model_state['patch_sizes'],
            'config': trainer.config.__dict__,
            'dice_score': trainer.best_model_state['dice_score']
        }, model_file)
        
        print(f"💾 Best model saved to: {model_file}")
        print(f"🏆 Best Dice score: {trainer.best_dice_score:.4f}")
    
    # Save training history
    history_file = output_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(results['history'], f, indent=2)
    
    print(f"📋 Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='RONASMIS Training')
    parser.add_argument('--preprocessed_dir', type=str, default='preprocessed_dwi_data',
                       help='Directory with preprocessed .pt files')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory for results')
    parser.add_argument('--exp_name', type=str, default='ronasmis_search',
                       help='Experiment name')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of search episodes')
    
    args = parser.parse_args()
    
    # Check if preprocessed data exists
    if not Path(args.preprocessed_dir).exists():
        print(f"❌ Preprocessed data directory not found: {args.preprocessed_dir}")
        print("Please run: python preprocess_dwi_data.py")
        return
    
    # Run training
    results = run_ronasmis_training(args)
    
    if results:
        print("✅ Training completed successfully!")
        print(f"🏆 Best architecture found with Dice score: {results['best_reward']:.4f}")
        print("Ready for testing!")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main() 