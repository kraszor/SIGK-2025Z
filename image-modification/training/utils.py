import torch
import time
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, criterion, optimizer, device, epoch=None):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (x, y) in tqdm(enumerate(loader)):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, "
                       f"Batch Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} Training - Avg Loss: {avg_loss:.6f}, "
                   f"Time: {epoch_time:.2f}s")
    
    return avg_loss

def validate(model, loader, criterion, device, epoch=None):
    model.eval()
    total_loss = 0
    batch_count = 0
  
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch} Validation - Loss: {avg_loss:.6f}")
    
    return avg_loss