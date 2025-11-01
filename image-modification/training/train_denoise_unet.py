import torch
from torch.utils.data import DataLoader
import time
from datetime import datetime
from utils import set_seed, train_epoch
from models.image_denoising import UNet
from data.dataset import ImageDataset
from loss_functions import CombinedLoss, CharbonnierLoss, PerceptualLoss


def train_model(data, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = data
    
    model = UNet(
        in_channels=3,
        init_features=128,
        use_bnorm=True,
    ).to(device)
    
    criterion = CombinedLoss(
        charbonnier_loss=CharbonnierLoss(),
        perceptual_loss=PerceptualLoss().to(device),
        mse_weight=1.0,
        charbonnier_weight=0.5,
        perceptual_weight=0.1
    ).to(device)
    
    print("Loss Configuration:")
    print("Combined Loss: MSE(1.0) + Charbonnier(0.5) + Perceptual(0.1)")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    print("Optimizer: Adam, LR: 1e-4, Weight Decay: 1e-5")
    print("-" * 80)

    best_val_loss = float("inf")
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Estimated remaining: {(total_time / (epoch + 1) * (epochs - epoch - 1)):.2f}s")

        print("-" * 40)

    total_training_time = time.time() - training_start_time
    print(f"Best validation loss: {best_val_loss:.6f}")
    model_filename = "best_model_unet_init"
    model_filename += ".pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Final model saved as: {model_filename}")

    print("="*80)
    print("TRAINING COMPLETED")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} min)")
    print(f"Average time per epoch: {total_training_time/epochs:.2f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final model saved as: {model_filename}")
    
    return best_val_loss


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train U-Net with comprehensive logging for Slurm")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    print("="*80)
    print("U-Net Training Experiment")
    print(f"Epochs: {args.epochs}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("="*80)
    
    train_data_dir = "./DIV2K_train_HR"
    val_data_dir = "./DIV2K_val_HR"
    set_seed(42)

    print("Loading dataset...")
    train_dataset = ImageDataset(root=train_data_dir, noise_level=0.1)
    print(f"Total images found: {len(train_dataset)}")
    val_dataset = ImageDataset(root=val_data_dir, noise_level=0.1)

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
    
    best_training_loss = train_model(train_loader, args.init_features, args.epochs)

    experiment_total_time = time.time() - experiment_start_time
    
    print("="*80)
    print("EXPERIMENT COMPLETED")
    print(f"Total experiment time: {experiment_total_time:.2f}s ({experiment_total_time/60:.1f} min)")
    print(f"Best training loss: {best_training_loss:.6f}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
