"""
Training script for U-Net for Slurm execution
Uses Combined Loss (MSE + Charbonnier + Perceptual)
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import random
import logging
import time
import json
from datetime import datetime
from models.image_denoising import UNet
from data.dataset import ImageDataset
from loss_functions import CombinedLoss, CharbonnierLoss, PerceptualLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(init_features: int, slurm_job_id: str = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if slurm_job_id:
        log_filename = f"unet_training_init{init_features}_job{slurm_job_id}_{timestamp}.log"
    else:
        log_filename = f"unet_training_init{init_features}_{timestamp}.log"
    
    log_filepath = os.path.join("logs", log_filename)
    
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info(f"Starting U-Net training with init_features={init_features}")
    if slurm_job_id:
        logger.info(f"Slurm Job ID: {slurm_job_id}")
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_PROCID', 'SLURM_SUBMIT_DIR', 'SLURM_JOB_NAME']
    for var in slurm_vars:
        if var in os.environ:
            logger.info(f"{var}: {os.environ[var]}")
    
    logger.info("="*80)
    
    return logger, log_filepath


def train_epoch(model, loader, criterion, optimizer, device, logger=None, epoch=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if logger and batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, "
                       f"Batch Loss: {loss.item():.6f}")

    avg_loss = total_loss / (batch_idx + 1)
    epoch_time = time.time() - start_time
    
    if logger:
        logger.info(f"Epoch {epoch} Training - Avg Loss: {avg_loss:.6f}, "
                   f"Time: {epoch_time:.2f}s")
    
    return avg_loss

def validate(model, loader, criterion, device, logger=None, epoch=None):
    model.eval()
    total_loss = 0
    batch_count = 0
    psnr_total = 0.0
    start_time = time.time()
  
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            mse = torch.mean((outputs - y) ** 2)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            psnr_total += psnr.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    avg_psnr = psnr_total / batch_count
    val_time = time.time() - start_time
    
    if logger:
        logger.info(f"Epoch {epoch} Validation - Loss: {avg_loss:.6f}, "
                   f"PSNR: {avg_psnr:.2f} dB, Time: {val_time:.2f}s")
    
    return avg_loss, avg_psnr

def train_model(data, init_features, epochs=10):
    try:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
        
        logger, log_filepath = setup_logging(init_features, slurm_job_id)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader = data
        
        logger.info("Dataset Information:")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        
        model = UNet(
            in_channels=3,
            init_features=init_features,
            use_bnorm=True,
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("Model Configuration:")
        logger.info(f"Init features: {init_features}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {total_params * 4 / 1e6:.2f} MB (float32)")
        
        criterion = CombinedLoss(
            charbonnier_loss=CharbonnierLoss(),
            perceptual_loss=PerceptualLoss().to(device),
            mse_weight=1.0,
            charbonnier_weight=0.5,
            perceptual_weight=0.1
        ).to(device)
        
        logger.info("Loss Configuration:")
        logger.info("Combined Loss: MSE(1.0) + Charbonnier(0.5) + Perceptual(0.1)")
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        logger.info("Optimizer: Adam, LR: 1e-4, Weight Decay: 1e-5")
        logger.info("-" * 80)

        best_val_loss = float("inf")
        best_val_psnr = 0.0
        training_start_time = time.time()
        
        training_metrics = {
            'init_features': init_features,
            'slurm_job_id': slurm_job_id,
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'val_psnrs': [],
            'epoch_times': [],
            'best_val_loss': None,
            'best_val_psnr': None,
            'total_training_time': None,
            'model_params': total_params,
            'log_file': log_filepath
        }
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch + 1)
            
            val_loss, val_psnr = validate(model, val_loader, criterion, device, logger, epoch + 1)
            
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time
            
            training_metrics['epochs'].append(epoch + 1)
            training_metrics['train_losses'].append(train_loss)
            training_metrics['val_losses'].append(val_loss)
            training_metrics['val_psnrs'].append(val_psnr)
            training_metrics['epoch_times'].append(epoch_time)
            
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Val PSNR: {val_psnr:.2f} dB")
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            logger.info(f"  Total Time: {total_time:.2f}s")
            logger.info(f"  Estimated remaining: {(total_time / (epoch + 1) * (epochs - epoch - 1)):.2f}s")
            
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_val_psnr = val_psnr
                model_filename = f"best_model_unet_init{init_features}"
                if slurm_job_id:
                    model_filename += f"_job{slurm_job_id}"
                model_filename += ".pth"
                torch.save(model.state_dict(), model_filename)
                logger.info(f"  *** NEW BEST MODEL! Improvement: {improvement:.6f}")
                logger.info(f"  Saved as: {model_filename}")
            else:
                logger.info(f"  No improvement (best: {best_val_loss:.6f})")
            
            logger.info("-" * 40)
        
        total_training_time = time.time() - training_start_time
        training_metrics['best_val_loss'] = best_val_loss
        training_metrics['best_val_psnr'] = best_val_psnr
        training_metrics['total_training_time'] = total_training_time
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED")
        logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} min)")
        logger.info(f"Average time per epoch: {total_training_time/epochs:.2f}s")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Best validation PSNR: {best_val_psnr:.2f} dB")
        logger.info(f"Final model saved as: {model_filename}")
        logger.info(f"Log saved as: {log_filepath}")
        
        metrics_filename = f"training_metrics_init{init_features}"
        if slurm_job_id:
            metrics_filename += f"_job{slurm_job_id}"
        metrics_filename += ".json"
        
        for key in training_metrics:
            if isinstance(training_metrics[key], list):
                training_metrics[key] = [float(x) if hasattr(x, 'item') else x for x in training_metrics[key]]
        
        with open(metrics_filename, 'w') as f:
            import json
            json.dump(training_metrics, f, indent=2)
        
        logger.info(f"Training metrics saved as: {metrics_filename}")
        logger.info("="*80)
        
        return best_val_loss, best_val_psnr
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in training: {str(e)}")
            logger.error("Traceback:", exc_info=True)
        print(f"Error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train U-Net with comprehensive logging for Slurm")
    parser.add_argument("--init_features", type=int, default=10, help="Initial number of features")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    print("="*80)
    print("U-Net Training Experiment")
    print(f"Init Features: {args.init_features}")
    print(f"Epochs: {args.epochs}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if 'SLURM_JOB_ID' in os.environ:
        print(f"Slurm Job ID: {os.environ['SLURM_JOB_ID']}")
        print(f"Slurm Job Name: {os.environ.get('SLURM_JOB_NAME', 'N/A')}")
    
    print("="*80)
    
    data_dir = "./DIV2K_train_HR"
    set_seed(42)

    print("Loading dataset...")
    full_dataset = ImageDataset(root=data_dir, noise_level=0.1)
    print(f"Total images found: {len(full_dataset)}")
    print(f"Sample image paths: {full_dataset.image_paths[:3]}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Train set size: {train_size}")
    print(f"Validation set size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print("Starting training...")
    experiment_start_time = time.time()
    
    best_val_loss, best_val_psnr = train_model((train_loader, val_loader), args.init_features, args.epochs)

    experiment_total_time = time.time() - experiment_start_time
    
    print("="*80)
    print("EXPERIMENT COMPLETED")
    print(f"Total experiment time: {experiment_total_time:.2f}s ({experiment_total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check logs/ directory for detailed training logs")
    print("="*80)
