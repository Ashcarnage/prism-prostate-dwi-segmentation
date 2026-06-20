#!/usr/bin/env python3
"""
RONASMIS Model Testing and Inference Script
Tests the trained RONASMIS model and exports results in .nrrd format
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import nrrd
from pathlib import Path
import time
from typing import Dict, Tuple, Optional

# Import RONASMIS components
from ronasmis import (
    RONASMISConfig, ChildNetwork, DiceLoss
)

def setup_gpu_optimization():
    """Setup GPU optimizations"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - using CPU")
        return torch.device('cpu')
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 GPU: {gpu_name}")
    
    return device

class RONASMISInference:
    """Class for RONASMIS model inference and testing"""
    
    def __init__(self, model_path: str = "experiments/ronasmis_search/best_ronasmis_model.pt"):
        self.model_path = Path(model_path)
        self.device = setup_gpu_optimization()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model checkpoint
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.architecture = self.checkpoint["architecture"]
        self.patch_sizes = self.checkpoint["patch_sizes"]
        self.dice_score = self.checkpoint["dice_score"]
        
        print(f"🏆 Best Architecture (Dice: {self.dice_score:.4f}):")
        print(f"   Patch HW: {self.architecture['patch_hw']} -> {self.patch_sizes['hw_sizes'][self.architecture['patch_hw']]}")
        print(f"   Patch D: {self.architecture['patch_d']} -> {self.patch_sizes['d_sizes'][self.architecture['patch_d']]}")
        
        activation_names = ['relu', 'leaky_relu', 'elu']
        activation_idx = self.architecture.get('activation_function', 0)
        print(f"   Activation: {activation_names[activation_idx]}")
        print(f"   Skip connections: {self.architecture['skip_connections']}")
        
        # Create config from saved data
        self.config = RONASMISConfig()
        if 'config' in self.checkpoint:
            # Update config with saved values
            for key, value in self.checkpoint['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
    def create_model(self) -> nn.Module:
        """Create and load the trained model"""
        model = ChildNetwork(
            self.config, 
            self.architecture, 
            self.patch_sizes,
            in_channels=1
        ).to(self.device)
        
        # Load the trained weights
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def test_inference(self, model: nn.Module, preprocessed_dir: str = "preprocessed_dwi_data") -> Dict:
        """Test model inference on preprocessed data"""
        
        preprocessed_path = Path(preprocessed_dir)
        if not preprocessed_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {preprocessed_dir}")
        
        # Load manifest
        manifest_path = preprocessed_path / "preprocessing_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        files = manifest['files']
        print(f"🧪 Testing on {len(files)} samples...")
        
        results = {
            'predictions': [],
            'dice_scores': [],
            'file_info': [],
            'inference_times': []
        }
        
        model.eval()
        
        with torch.no_grad():
            for i, file_info in enumerate(files):
                start_time = time.time()
                
                # Load preprocessed data
                data = torch.load(file_info['file'])
                image = data['image'].to(self.device)
                label = data['label'].to(self.device)
                
                # Store original size for final output
                original_size = image.shape[1:]  # Remove channel dimension
                
                # Add batch dimension
                image_batch = image.unsqueeze(0)
                label_batch = label.unsqueeze(0)
                
                # Resize to architecture's expected patch size
                target_hw = self.patch_sizes['hw_sizes'][self.architecture['patch_hw']]
                target_d = self.patch_sizes['d_sizes'][self.architecture['patch_d']]
                target_size = (target_d, target_hw, target_hw)
                
                if image_batch.shape[2:] != target_size:
                    image_batch = F.interpolate(
                        image_batch, size=target_size, 
                        mode='trilinear', align_corners=False
                    )
                    # Also resize labels to match for dice calculation
                    label_batch_resized = F.interpolate(
                        label_batch, size=target_size, 
                        mode='trilinear', align_corners=False
                    )
                else:
                    label_batch_resized = label_batch
                
                # Inference
                with torch.amp.autocast('cuda'):
                    outputs = model(image_batch)
                    
                    # Handle tuple output
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Apply sigmoid and threshold
                    predictions = torch.sigmoid(outputs) > 0.5
                
                # Calculate dice score on resized data (same size as predictions)
                dice = self._calculate_dice_score(predictions, label_batch_resized)
                
                # Resize predictions back to original size for saving
                if predictions.shape[2:] != original_size:
                    predictions_original = F.interpolate(
                        predictions.float(), size=original_size,
                        mode='trilinear', align_corners=False
                    ) > 0.5
                else:
                    predictions_original = predictions
                
                inference_time = time.time() - start_time
                
                # Store results (use original-sized predictions for export)
                results['predictions'].append(predictions_original.cpu())
                results['dice_scores'].append(dice)
                results['file_info'].append({
                    'original_file': file_info['file'],
                    'patient_info': file_info['patient_info'],
                    'original_shape': list(original_size),
                    'prediction_shape': list(predictions_original.shape)
                })
                results['inference_times'].append(inference_time)
                
                if i % 5 == 0 or i < 3:
                    print(f"   Sample {i+1}: Dice={dice:.4f}, Time={inference_time:.2f}s")
        
        avg_dice = np.mean(results['dice_scores'])
        avg_time = np.mean(results['inference_times'])
        
        print(f"\n📊 Test Results:")
        print(f"   Average Dice Score: {avg_dice:.4f}")
        print(f"   Average Inference Time: {avg_time:.3f}s")
        print(f"   Total Samples: {len(results['dice_scores'])}")
        
        return results
    
    def export_results(self, results: Dict, output_dir: str):
        """Export predictions and ground truth to .nrrd format"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"💾 Exporting {len(results['predictions'])} segmentations and ground truth to {output_dir}")
        
        for i, (pred, file_info, dice) in enumerate(zip(
            results['predictions'], 
            results['file_info'], 
            results['dice_scores']
        )):
            # Load original data to get ground truth
            data = torch.load(file_info['original_file'])
            ground_truth = data['label'].squeeze().cpu().numpy()
            
            # Get prediction as numpy array  
            prediction = pred.squeeze().cpu().numpy().astype(np.uint8)
            
            # Create filenames
            patient_info = file_info['patient_info']
            base_filename = f"{patient_info}_{i:03d}"
            
            pred_filename = f"prediction_{base_filename}_dice{dice:.3f}.nrrd"
            gt_filename = f"groundtruth_{base_filename}.nrrd"
            
            pred_path = output_path / pred_filename
            gt_path = output_path / gt_filename
            
            # Save prediction
            nrrd.write(str(pred_path), prediction)
            
            # Save ground truth  
            nrrd.write(str(gt_path), ground_truth.astype(np.uint8))
            
            if i < 3:  # Show first few for verification
                pred_shape = prediction.shape
                gt_shape = ground_truth.shape
                print(f"   ✓ {pred_filename}: {pred_shape}, Dice={dice:.4f}")
                print(f"   ✓ {gt_filename}: {gt_shape}")
        
        # Create summary
        summary = {
            'total_samples': len(results['predictions']),
            'average_dice_score': float(np.mean(results['dice_scores'])),
            'average_inference_time': float(np.mean(results['inference_times'])),
            'dice_scores': [float(d) for d in results['dice_scores']],
            'inference_times': results['inference_times'],
            'file_mapping': [
                {
                    'prediction_file': f"prediction_{info['patient_info']}_{i:03d}_dice{dice:.3f}.nrrd",
                    'groundtruth_file': f"groundtruth_{info['patient_info']}_{i:03d}.nrrd",
                    'patient_info': info['patient_info'],
                    'dice_score': float(dice),
                    'original_shape': info['original_shape'],
                    'prediction_shape': info['prediction_shape']
                }
                for i, (info, dice) in enumerate(zip(results['file_info'], results['dice_scores']))
            ]
        }
        
        # Save detailed summary
        summary_path = output_path / "inference_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Summary saved to: {summary_path}")
        print(f"🎯 Both predictions and ground truth ready for 3D Slicer comparison!")
        
        return summary
    
    def _calculate_dice_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate dice score between predictions and targets"""
        pred_binary = (predictions > 0.5).float()
        target_binary = (targets > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
            return dice.item()
        else:
            return 1.0  # Perfect score for empty regions

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RONASMIS Model Testing')
    parser.add_argument('--model_path', type=str, 
                       default='experiments/ronasmis_search/best_ronasmis_model.pt',
                       help='Path to trained model')
    parser.add_argument('--preprocessed_dir', type=str, 
                       default='preprocessed_dwi_data',
                       help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, 
                       default='segmentation_results',
                       help='Output directory for segmentations')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 RONASMIS Model Testing")
    print("=" * 60)
    
    try:
        # Initialize inference
        inference = RONASMISInference(args.model_path)
        
        # Create model
        model = inference.create_model()
        
        # Test inference
        results = inference.test_inference(model, args.preprocessed_dir)
        
        # Export results
        summary = inference.export_results(results, args.output_dir)
        
        print("\n🎉 Testing completed successfully!")
        print(f"📁 Results saved to: {summary['file_mapping'][0]['prediction_file']}")
        print("Ready for 3D Slicer visualization!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 