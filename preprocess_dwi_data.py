#!/usr/bin/env python3
"""
DWI Data Preprocessing for PRISM
Converts DWI dataset to optimized .pt files while preserving original dimensions
"""

import os
import json
import numpy as np
import torch
import nibabel as nib
import nrrd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class DWIPreprocessor:
    """Preprocessor for DWI data following PRISM specifications"""
    
    def __init__(self, data_dir: str, output_dir: str = "preprocessed_dwi_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_and_normalize_image(self, path: str) -> np.ndarray:
        """Load and normalize medical image following PRISM protocol"""
        if path.endswith('.nrrd'):
            data, _ = nrrd.read(path)
        else:
            nii_img = nib.load(path)
            data = nii_img.get_fdata()
        
        # Convert to float32
        data = data.astype(np.float32)
        
        # Handle 4D images (take first volume)
        if len(data.shape) == 4:
            data = data[..., 0]
        
        # Z-score normalization for images (as per PRISM)
        if data.std() > 0:
            data = (data - data.mean()) / data.std()
        
        return data
    
    def load_and_process_label(self, path: str) -> np.ndarray:
        """Load and process label following PRISM protocol"""
        if path.endswith('.nrrd'):
            data, _ = nrrd.read(path)
        else:
            nii_img = nib.load(path)
            data = nii_img.get_fdata()
        
        # Convert to float32
        data = data.astype(np.float32)
        
        # Handle 4D labels (take first volume)
        if len(data.shape) == 4:
            data = data[..., 0]
        
        # Ensure binary labels
        data = (data > 0).astype(np.float32)
        
        return data
    
    def preprocess_dataset(self, dataset_pairs_file: str = "dwi_dataset_pairs.json"):
        """Preprocess entire dataset into .pt files preserving original sizes"""
        
        # Load dataset pairs
        pairs_file = self.data_dir / dataset_pairs_file
        if not pairs_file.exists():
            pairs_file = Path(dataset_pairs_file)
        
        with open(pairs_file, 'r') as f:
            dataset_pairs = json.load(f)
        
        print(f"🔄 Preprocessing {len(dataset_pairs)} DWI pairs...")
        print(f"📁 Output directory: {self.output_dir}")
        
        processed_files = []
        
        for idx, pair in enumerate(tqdm(dataset_pairs, desc="Processing DWI data")):
            try:
                # Load image and label
                image_data = self.load_and_normalize_image(pair['image'])
                label_data = self.load_and_process_label(pair['label'])
                
                # Ensure both have same spatial dimensions
                if image_data.shape != label_data.shape:
                    print(f"⚠️  Shape mismatch for pair {idx}: image {image_data.shape}, label {label_data.shape}")
                    # Skip mismatched pairs
                    continue
                
                # Add channel dimension: [C, D, H, W]
                image_tensor = torch.from_numpy(image_data[np.newaxis, ...]).float()
                label_tensor = torch.from_numpy(label_data[np.newaxis, ...]).float()
                
                # Create output filename
                output_filename = f"dwi_sample_{idx:04d}.pt"
                output_path = self.output_dir / output_filename
                
                # Save as .pt file with metadata
                torch.save({
                    'image': image_tensor,
                    'label': label_tensor,
                    'original_image_path': pair['image'],
                    'original_label_path': pair['label'],
                    'shape': image_tensor.shape,
                    'patient_info': pair.get('patient_id', f'unknown_{idx}')
                }, output_path)
                
                processed_files.append({
                    'file': str(output_path),
                    'shape': list(image_tensor.shape),
                    'patient_info': pair.get('patient_id', f'unknown_{idx}')
                })
                
                if idx < 3:  # Print first few for verification
                    print(f"✓ Sample {idx}: {image_tensor.shape} -> {output_filename}")
                
            except Exception as e:
                print(f"❌ Error processing pair {idx}: {e}")
                continue
        
        # Save preprocessing manifest
        manifest = {
            'total_processed': len(processed_files),
            'files': processed_files,
            'preprocessing_info': {
                'normalization': 'z-score for images, binary for labels',
                'format': 'torch tensors with shape [C, D, H, W]',
                'original_sizes_preserved': True
            }
        }
        
        manifest_path = self.output_dir / "preprocessing_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Preprocessing complete!")
        print(f"📊 Processed {len(processed_files)} files")
        print(f"📋 Manifest saved to: {manifest_path}")
        
        return processed_files

def main():
    """Main preprocessing function"""
    preprocessor = DWIPreprocessor(
        data_dir="Extracted_DWI",
        output_dir="preprocessed_dwi_data"
    )
    
    processed_files = preprocessor.preprocess_dataset()
    
    if processed_files:
        print(f"🎉 Successfully preprocessed {len(processed_files)} DWI samples!")
        print("Ready for PRISM training!")
    else:
        print("❌ No files were successfully processed!")

if __name__ == "__main__":
    main() 