"""
Training script for DnCNN with Ray Tune
Uses Combined Loss (MSE + Charbonnier + Perceptual)
"""

import os
import torch
import numpy as np
from ray import tune
import random
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import tempfile
from torch.utils.data import DataLoader, random_split
from models.image_denoising import DnCNN
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (batch_idx + 1)
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    batch_count = 0
    psnr_total = 0.0
  
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
    return avg_loss, avg_psnr

def train_model_tune(config, data_dir, checkpoint_dir=None):
    try:
        import sys
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_dataset = ImageDataset(root=data_dir, noise_level=config["noise_level"])
        print(f"Dataset size: {len(full_dataset)}", file=sys.stderr)
        print(f"First few image paths: {full_dataset.image_paths[:3]}", file=sys.stderr)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        model = DnCNN(
            in_channels=3,
            num_layers=config["num_layers"],
            num_features=config["num_features"],
            kernel_size=config["kernel_size"],
            use_bnorm=config["use_bnorm"]
        ).to(device)

        criterion = CombinedLoss(
            charbonnier_loss=CharbonnierLoss(),
            perceptual_loss=PerceptualLoss(),
            mse_weight=config["mse_weight"],
            charbonnier_weight=config["charbonnier_weight"],
            perceptual_weight=config["perceptual_weight"]).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        best_val_loss = float("inf")
        for epoch in range(config["num_epochs"]):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_psnr = validate(model, val_loader, criterion, device)

            tune.report(metrics={"loss": val_loss, "psnr": val_psnr})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model_dncnn.pth")
    except Exception as e:
        print(f"Error in trial: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def tune_hyperparameters(data_dir, num_samples=5, max_num_epochs=10, noise_level: float = 0.1):
    config = {
        "num_layers": tune.choice([15, 17, 20]),
        "num_features": tune.choice([64, 96, 128]),
        "kernel_size": tune.choice([3, 5]),
        "use_bnorm": tune.choice([True, False]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([8, 16, 32]),
        "noise_level": noise_level,
        "mse_weight": tune.uniform(0.5, 2.0),
        "charbonnier_weight": tune.uniform(0.5, 2.0),
        "perceptual_weight": tune.uniform(0.05, 0.2),
        "num_epochs": tune.choice([8, 10, 12])
    }

    scheduler = ASHAScheduler(
        metric="psnr",
        mode="max",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2
    )

    reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "psnr", "epoch"],)

    result = tune.run(
        tune.with_parameters(train_model_tune, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="dncnn_tune",
        # resume="AUTO"  
    )

    best_trial = result.get_best_trial("psnr", "max", "last")
    print("\n✅ Najlepsza konfiguracja:")
    print(best_trial.config)
    print(f"Val PSNR: {best_trial.last_result['psnr']:.2f}")

    return best_trial


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import glob
    import random
    import math
    import os
    data_dir = "./DIV2K_train_HR"

    full_dataset = ImageDataset(root=data_dir, noise_level=0.1)
    print(full_dataset.image_paths)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

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

    from ray import tune
    best_trial = tune_hyperparameters(data_dir)

    best_config = best_trial.config
    print("\nNajlepsze hiperparametry:", best_config)
