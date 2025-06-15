#!/usr/bin/env python3
"""
Script to check parameters and model information from PyTorch .pt files
"""

import torch
import os
import sys
from pathlib import Path

def count_parameters(model_dict):
    """Count total parameters in model state dict"""
    total_params = 0
    trainable_params = 0
    
    for name, param in model_dict.items():
        if isinstance(param, torch.Tensor):
            param_count = param.numel()
            total_params += param_count
            print(f"{name:50} | {param_count:>12,} | {list(param.shape)}")
    
    return total_params

def analyze_checkpoint(checkpoint_path):
    """Analyze a PyTorch checkpoint file"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {checkpoint_path}")
    print(f"{'='*80}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get file size
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"File Size: {file_size:.2f} MB")
        
        # Check what's in the checkpoint
        print(f"\nCheckpoint Keys: {list(checkpoint.keys())}")
        
        # Try different keys that might contain the model
        model_keys = ['model', 'model_state_dict', 'state_dict', 'net']
        model_dict = None
        
        for key in model_keys:
            if key in checkpoint:
                model_dict = checkpoint[key]
                print(f"\nUsing model from key: '{key}'")
                break
        
        # If no specific model key, use the whole checkpoint
        if model_dict is None:
            # Check if checkpoint itself is the state dict
            if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                model_dict = checkpoint
                print(f"\nUsing entire checkpoint as model state dict")
            else:
                print("Could not find model state dict in checkpoint")
                return
        
        # Count parameters
        print(f"\n{'Parameter Name':50} | {'Count':>12} | {'Shape'}")
        print(f"{'-'*50} | {'-'*12} | {'-'*20}")
        
        total_params = count_parameters(model_dict)
        
        print(f"\n{'='*80}")
        print(f"TOTAL PARAMETERS: {total_params:,}")
        print(f"MODEL SIZE: {file_size:.2f} MB")
        print(f"PARAMS PER MB: {total_params/file_size:,.0f}")
        print(f"{'='*80}")
        
        # Try to get additional info
        if 'optimizer' in checkpoint:
            print(f"Optimizer: Present")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"Step: {checkpoint['step']}")
        if 'best_val_loss' in checkpoint:
            print(f"Best Val Loss: {checkpoint['best_val_loss']}")
            
    except Exception as e:
        print(f"Error analyzing {checkpoint_path}: {str(e)}")

def main():
    """Main function to analyze checkpoints"""
    
    # Base directory for checkpoints
    base_dir = "/ssd-data1/sq2023/belajar/checkpoints"
    
    # Model directories to check
    model_dirs = [
        "baseline",
        "wordpiece", 
        "wordpiece_linked",
        "wordpiece_gated_linked",
        "wordpiece_kg_gated_linked",
        "castle",
        "castle_improved"
    ]
    
    print("PYTORCH MODEL PARAMETER ANALYSIS")
    print("="*80)
    
    # Summary table
    summary = []
    
    for model_dir in model_dirs:
        checkpoint_path = os.path.join(base_dir, model_dir, "checkpoint_best.pt")
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Get model dict
                model_keys = ['model', 'model_state_dict', 'state_dict', 'net']
                model_dict = None
                
                for key in model_keys:
                    if key in checkpoint:
                        model_dict = checkpoint[key]
                        break
                
                if model_dict is None and any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                    model_dict = checkpoint
                
                if model_dict:
                    total_params = sum(param.numel() for param in model_dict.values() if isinstance(param, torch.Tensor))
                    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
                    
                    summary.append({
                        'model': model_dir,
                        'params': total_params,
                        'size_mb': file_size,
                        'path': checkpoint_path
                    })
                
            except Exception as e:
                print(f"Error with {model_dir}: {str(e)}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Model Name':25} | {'Parameters':>15} | {'Size (MB)':>10} | {'Params/MB':>12}")
    print(f"{'-'*25} | {'-'*15} | {'-'*10} | {'-'*12}")
    
    for item in sorted(summary, key=lambda x: x['params']):
        params_per_mb = item['params'] / item['size_mb']
        print(f"{item['model']:25} | {item['params']:>15,} | {item['size_mb']:>10.2f} | {params_per_mb:>12,.0f}")
    
    # Detailed analysis if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        for item in summary:
            analyze_checkpoint(item['path'])

if __name__ == "__main__":
    main()