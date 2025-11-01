
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air.checkpoint import Checkpoint
import tempfile

from models.image_sr import VDSR
from data.dataset import ImageDataset


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_vdsr(config, data_dir: str, checkpoint_dir: str = None):
    
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VDSR(
        in_channels=config["in_channels"],
        num_layers=config["num_layers"],
        num_features=config["num_features"]
    ).to(device)
    
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    full_dataset = ImageDataset(
        root=data_dir,
        input_size=(config["input_size"], config["input_size"])
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for lr_img, hr_img in train_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            optimizer.zero_grad()
            output = model(lr_img)
            loss = criterion(output, hr_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        psnr_total = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                output = model(lr_img)
                loss = criterion(output, hr_img)
                val_loss += loss.item()
                mse = torch.mean((output - hr_img) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                psnr_total += psnr.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = psnr_total / len(val_loader)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint_path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "psnr": avg_psnr
            }, checkpoint_path)
            
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            session.report({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "psnr": avg_psnr,
                "epoch": epoch
            }, checkpoint=checkpoint)


def main(data_dir: str, num_samples: int = 10, num_epochs: int = 50, input_size: int = 32):
    
    set_seed(42)
    
    config = {
        "in_channels": 3,
        "num_layers": tune.choice([15, 20, 25]),
        "num_features": tune.choice([64, 96, 128]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([8, 16, 32]),
        "input_size": input_size,
        "num_epochs": num_epochs
    }
    
    scheduler = ASHAScheduler(
        metric="psnr",
        mode="max",
        max_t=num_epochs,
        grace_period=5,
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        metric_columns=["train_loss", "val_loss", "psnr", "epoch"],
        max_report_frequency=30
    )
    
    result = tune.run(
        tune.with_parameters(train_vdsr, data_dir=data_dir),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="vdsr_tune",
        local_dir="./ray_results",
        checkpoint_freq=5,
        keep_checkpoints_num=3,
        checkpoint_score_attr="loss"
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("\n" + "="*60)
    print("Best trial config:")
    print("="*60)
    for key, value in best_trial.config.items():
        print(f"{key:25s}: {value}")
    print(f"\nBest PSNR: {best_trial.last_result['psnr']:.2f} dB")
    print(f"Best val loss: {best_trial.last_result['val_loss']:.4f}")
    print("="*60)
    best_checkpoint_dir = best_trial.checkpoint.to_directory()
    best_checkpoint_path = os.path.join(best_checkpoint_dir, "checkpoint.pt")
    checkpoint = torch.load(best_checkpoint_path)
    final_model_path = "./best_vdsr_model.pth"
    torch.save(checkpoint["model_state_dict"], final_model_path)
    print(f"\nBest model saved to: {final_model_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VDSR with Ray Tune")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of trials (default: 10)")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs per trial (default: 50)")
    
    args = parser.parse_args()
    
    main(args.data_dir, args.num_samples, args.num_epochs)
