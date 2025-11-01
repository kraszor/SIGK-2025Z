#!/usr/bin/env python3
"""
Script to run multiple U-Net training experiments with different init_features values
"""

import subprocess
import sys
import time
from datetime import datetime

def run_experiment(init_features):
    """Run a single experiment with specified init_features"""
    print(f"\n{'='*60}")
    print(f"Starting experiment with init_features={init_features}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the training script
        cmd = [sys.executable, "train_unet.py", "--init_features", str(init_features)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✓ Experiment completed successfully in {duration:.1f}s")
            print(f"Check logs/ directory for detailed logs")
        else:
            print(f"✗ Experiment failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode == 0, duration
        
    except subprocess.TimeoutExpired:
        print(f"✗ Experiment timed out after 1 hour")
        return False, 3600
    except Exception as e:
        print(f"✗ Experiment failed with exception: {e}")
        return False, time.time() - start_time

def main():
    """Run experiments with different init_features values"""
    # Define the init_features values to test
    init_features_list = [8, 16, 32, 64]
    
    print("U-Net Training Experiments")
    print(f"Testing init_features values: {init_features_list}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    total_start_time = time.time()
    
    for init_features in init_features_list:
        success, duration = run_experiment(init_features)
        results.append({
            'init_features': init_features,
            'success': success,
            'duration': duration
        })
        
        # Brief pause between experiments
        time.sleep(5)
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Results:")
    print(f"{'Init Features':<15} {'Status':<10} {'Duration (s)':<15}")
    print("-" * 40)
    
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{result['init_features']:<15} {status:<10} {result['duration']:<15.1f}")
    
    print(f"\n{'='*80}")
    
    # Count successes
    successful = sum(1 for r in results if r['success'])
    print(f"Successful experiments: {successful}/{len(results)}")
    
    if successful > 0:
        print("\nCheck the following for results:")
        print("- logs/ directory for detailed training logs")
        print("- best_model_unet_init*.pth files for saved models")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
