#!/usr/bin/env python3
"""
Script to analyze and compare U-Net training logs
"""

import os
import re
import glob
from datetime import datetime
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """Parse a training log file and extract key metrics"""
    if not os.path.exists(log_path):
        return None
    
    data = {
        'init_features': None,
        'total_params': None,
        'model_size_mb': None,
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'val_psnrs': [],
        'epoch_times': [],
        'best_val_loss': None,
        'best_val_psnr': None,
        'total_training_time': None
    }
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract init_features
        match = re.search(r'Init features: (\d+)', content)
        if match:
            data['init_features'] = int(match.group(1))
        
        # Extract model info
        match = re.search(r'Total parameters: ([\d,]+)', content)
        if match:
            data['total_params'] = int(match.group(1).replace(',', ''))
        
        match = re.search(r'Model size: ([\d.]+) MB', content)
        if match:
            data['model_size_mb'] = float(match.group(1))
        
        # Extract epoch data
        epoch_pattern = r'Epoch (\d+) Summary:\s+Train Loss: ([\d.]+)\s+Val Loss: ([\d.]+)\s+Val PSNR: ([\d.]+) dB\s+Epoch Time: ([\d.]+)s'
        for match in re.finditer(epoch_pattern, content):
            epoch, train_loss, val_loss, val_psnr, epoch_time = match.groups()
            data['epochs'].append(int(epoch))
            data['train_losses'].append(float(train_loss))
            data['val_losses'].append(float(val_loss))
            data['val_psnrs'].append(float(val_psnr))
            data['epoch_times'].append(float(epoch_time))
        
        # Extract best results
        match = re.search(r'Best validation loss: ([\d.]+)', content)
        if match:
            data['best_val_loss'] = float(match.group(1))
        
        match = re.search(r'Best validation PSNR: ([\d.]+) dB', content)
        if match:
            data['best_val_psnr'] = float(match.group(1))
        
        # Extract total training time
        match = re.search(r'Total training time: ([\d.]+)s', content)
        if match:
            data['total_training_time'] = float(match.group(1))
        
        return data
    
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None

def analyze_logs():
    """Analyze all log files in the logs directory"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print(f"Logs directory '{logs_dir}' not found!")
        return
    
    log_files = glob.glob(os.path.join(logs_dir, "unet_training_init*.log"))
    
    if not log_files:
        print(f"No U-Net training log files found in '{logs_dir}'")
        return
    
    print(f"Found {len(log_files)} log files")
    print("-" * 60)
    
    experiments = []
    
    for log_file in sorted(log_files):
        print(f"Analyzing: {os.path.basename(log_file)}")
        data = parse_log_file(log_file)
        
        if data and data['init_features'] is not None:
            experiments.append(data)
            print(f"  Init Features: {data['init_features']}")
            print(f"  Total Params: {data['total_params']:,}")
            print(f"  Model Size: {data['model_size_mb']:.2f} MB")
            print(f"  Best Val Loss: {data['best_val_loss']:.6f}")
            print(f"  Best Val PSNR: {data['best_val_psnr']:.2f} dB")
            print(f"  Training Time: {data['total_training_time']:.1f}s")
            print()
    
    if not experiments:
        print("No valid experiment data found!")
        return
    
    # Create comparison table
    print("COMPARISON TABLE")
    print("=" * 100)
    
    header = f"{'Init Feat':<10} {'Params':<12} {'Size(MB)':<10} {'Best Loss':<12} {'Best PSNR':<12} {'Time(s)':<10}"
    print(header)
    print("-" * 100)
    
    for exp in sorted(experiments, key=lambda x: x['init_features']):
        row = f"{exp['init_features']:<10} {exp['total_params']:<12,} {exp['model_size_mb']:<10.2f} "
        row += f"{exp['best_val_loss']:<12.6f} {exp['best_val_psnr']:<12.2f} {exp['total_training_time']:<10.1f}"
        print(row)
    
    print("=" * 100)
    
    # Find best performing model
    best_exp = min(experiments, key=lambda x: x['best_val_loss'])
    best_psnr_exp = max(experiments, key=lambda x: x['best_val_psnr'])
    
    print(f"\nBest Loss: Init Features = {best_exp['init_features']} (Loss: {best_exp['best_val_loss']:.6f})")
    print(f"Best PSNR: Init Features = {best_psnr_exp['init_features']} (PSNR: {best_psnr_exp['best_val_psnr']:.2f} dB)")
    
    # Generate plots if possible
    try:
        generate_plots(experiments)
    except ImportError:
        print("\nNote: Install matplotlib to generate comparison plots")
    except Exception as e:
        print(f"\nError generating plots: {e}")

def generate_plots(experiments):
    """Generate comparison plots"""
    if len(experiments) < 2:
        print("Need at least 2 experiments for comparison plots")
        return
    
    # Prepare data for plotting
    init_features = [exp['init_features'] for exp in experiments]
    best_losses = [exp['best_val_loss'] for exp in experiments]
    best_psnrs = [exp['best_val_psnr'] for exp in experiments]
    model_sizes = [exp['model_size_mb'] for exp in experiments]
    training_times = [exp['total_training_time'] for exp in experiments]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Best Loss vs Init Features
    ax1.plot(init_features, best_losses, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Initial Features')
    ax1.set_ylabel('Best Validation Loss')
    ax1.set_title('Best Validation Loss vs Initial Features')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best PSNR vs Init Features
    ax2.plot(init_features, best_psnrs, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Initial Features')
    ax2.set_ylabel('Best PSNR (dB)')
    ax2.set_title('Best PSNR vs Initial Features')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model Size vs Init Features
    ax3.plot(init_features, model_sizes, 'o-', color='red', linewidth=2, markersize=8)
    ax3.set_xlabel('Initial Features')
    ax3.set_ylabel('Model Size (MB)')
    ax3.set_title('Model Size vs Initial Features')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Time vs Init Features
    ax4.plot(init_features, training_times, 'o-', color='orange', linewidth=2, markersize=8)
    ax4.set_xlabel('Initial Features')
    ax4.set_ylabel('Training Time (s)')
    ax4.set_title('Training Time vs Initial Features')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"unet_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nComparison plots saved as: {plot_filename}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    analyze_logs()
