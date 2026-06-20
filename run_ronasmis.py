#!/usr/bin/env python3
"""
Quick Start Script for RONASMIS
Simplified interface with sensible defaults for fast experimentation
"""

import os
import sys
import torch
from termcolor import colored

def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        print(colored("❌ CUDA not available. RONASMIS requires GPU for efficient training.", "red"))
        return False
    
    device_name = torch.cuda.get_device_name()
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(colored(f"✅ GPU Found: {device_name}", "green"))
    print(colored(f"✅ GPU Memory: {memory_gb:.1f} GB", "green"))
    
    if memory_gb < 8:
        print(colored("⚠️  Warning: Less than 8GB GPU memory. Consider reducing batch size.", "yellow"))
    
    return True

def check_dataset():
    """Check if DWI dataset exists"""
    data_dir = "Extracted_DWI"
    if not os.path.exists(data_dir):
        print(colored(f"❌ Dataset directory '{data_dir}' not found.", "red"))
        print(colored("   Please ensure your DWI dataset is in the 'Extracted_DWI' folder.", "yellow"))
        return False
    
    # Count DWI files
    dwi_files = [f for f in os.listdir(data_dir) if 'DWI' in f and f.endswith(('.nii.gz', '.nrrd'))]
    print(colored(f"✅ Found {len(dwi_files)} DWI files in dataset", "green"))
    
    if len(dwi_files) < 10:
        print(colored("⚠️  Warning: Few DWI files found. Results may be limited.", "yellow"))
    
    return True

def run_quick_search():
    """Run a quick architecture search with minimal parameters"""
    print(colored("\n🚀 Starting Quick RONASMIS Search", "magenta"))
    print(colored("   Episodes: 25, Networks per episode: 8", "cyan"))
    
    cmd = [
        "python", "train_ronasmis.py",
        "--total_episodes", "25",
        "--child_networks_per_episode", "8",
        "--child_epochs_per_episode", "2",
        "--batch_size", "2",
        "--output_dir", "ronasmis_quick",
        "--experiment_name", "quick_search"
    ]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print(colored("✅ Quick search completed!", "green"))
    except subprocess.CalledProcessError as e:
        print(colored(f"❌ Error during search: {e}", "red"))

def run_full_search():
    """Run a full architecture search with optimal parameters"""
    print(colored("\n🎯 Starting Full RONASMIS Search", "magenta"))
    print(colored("   Episodes: 75, Networks per episode: 12", "cyan"))
    print(colored("   This will take approximately 12-18 hours on RTX 5090", "cyan"))
    
    cmd = [
        "python", "train_ronasmis.py",
        "--total_episodes", "75",
        "--child_networks_per_episode", "12",
        "--child_epochs_per_episode", "3",
        "--batch_size", "2",
        "--use_wandb",
        "--output_dir", "ronasmis_full",
        "--experiment_name", "full_search"
    ]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print(colored("✅ Full search completed!", "green"))
    except subprocess.CalledProcessError as e:
        print(colored(f"❌ Error during search: {e}", "red"))

def test_components():
    """Test RONASMIS components"""
    print(colored("\n🔧 Testing RONASMIS Components", "blue"))
    
    try:
        from ronasmis import RONASMISConfig, SearchSpace, LSTMController, ChildNetwork
        
        # Test configuration
        config = RONASMISConfig()
        print(colored("✅ Configuration loaded", "green"))
        
        # Test search space
        search_space = SearchSpace(config)
        sample_arch = search_space.sample_architecture()
        print(colored(f"✅ Search space working: {sample_arch}", "green"))
        
        # Test controller
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controller = LSTMController(config, search_space).to(device)
        
        with torch.no_grad():
            decisions, log_prob, entropy = controller()
            print(colored(f"✅ Controller working: {decisions}", "green"))
        
        # Test child network
        child_net = ChildNetwork(sample_arch, in_channels=1, out_channels=1)
        dummy_input = torch.randn(1, 1, 32, 32, 8)  # Small test input
        
        with torch.no_grad():
            output = child_net(dummy_input)
            print(colored(f"✅ Child network working: output shape {output.shape}", "green"))
        
        print(colored("✅ All components working correctly!", "green"))
        return True
        
    except Exception as e:
        print(colored(f"❌ Component test failed: {e}", "red"))
        return False

def main():
    """Main interface"""
    print(colored("=" * 60, "magenta"))
    print(colored("RONASMIS Quick Start", "magenta"))
    print(colored("Resource Optimized NAS for 3D Medical Segmentation", "magenta"))
    print(colored("=" * 60, "magenta"))
    
    # Check system requirements
    if not check_gpu():
        return
    
    if not check_dataset():
        return
    
    # Test components
    if not test_components():
        print(colored("⚠️  Component test failed. Please check installation.", "yellow"))
        return
    
    # User choice
    print(colored("\n📋 Choose an option:", "cyan"))
    print("1. Quick Search (25 episodes, ~2-4 hours)")
    print("2. Full Search (75 episodes, ~12-18 hours)")
    print("3. Test Components Only")
    print("4. Exit")
    
    choice = input(colored("\nEnter your choice (1-4): ", "yellow"))
    
    if choice == "1":
        run_quick_search()
    elif choice == "2":
        # Confirm full search
        confirm = input(colored("\nFull search will take 12-18 hours. Continue? (y/N): ", "yellow"))
        if confirm.lower() == 'y':
            run_full_search()
        else:
            print("Full search cancelled.")
    elif choice == "3":
        print(colored("✅ Component test completed successfully!", "green"))
    elif choice == "4":
        print(colored("Goodbye! 👋", "cyan"))
    else:
        print(colored("Invalid choice. Please run again.", "red"))

if __name__ == "__main__":
    main() 