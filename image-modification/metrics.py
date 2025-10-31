import math
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import lpips
from torch.utils.data import DataLoader

def SNE(original, generated):
    mse = np.mean((original - generated) ** 2)
    return mse

def evaluate_SNE(model: nn.Module, data: DataLoader, device: str):
    model.eval()
    total_score = 0
    num_batches = 0
    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = modified_img.to(device), original_img.to(device)
            generated_img = model(modified_img)  

            score = SNE(generated_img, original_img) 

            total_score += score.item()
            num_batches += 1
    average_score = total_score / num_batches
    print(f"Average SNE: {average_score:.6f}")
    return average_score

def PSNR(original, generated):
    mse = SNE(original, generated)
    if(mse == 0):
        return float("inf")
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def evaluate_PSNR(model: nn.Module, data: DataLoader, device: str):
    model.eval()
    total_score = 0
    num_batches = 0

    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = modified_img.to(device), original_img.to(device)
            generated_img = model(modified_img)  

            score = PSNR(generated_img, original_img) 

            total_score += score.item()
            num_batches += 1
    average_score = total_score / num_batches
    print(f"Average PSNR: {average_score:.2f} dB")
    return average_score   

def evaluate_SSIM(model: nn.Module, data: DataLoader, device: str):
    model.eval()
    total_score = 0
    num_batches = 0
    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = modified_img.to(device), original_img.to(device)
            generated_img = model(modified_img)  

            score = SSIM(generated_img, original_img) 

            total_score += score.item()
            num_batches += 1
    average_score = total_score / num_batches
    print(f"Average SSIM: {average_score:.4f}")
    return average_score



def SSIM(original, generated) :
    score, dif = ssim(original, generated, full=True)
    return score


def evaluate_LPIPS(model: nn.Module, data: DataLoader, device: str):

    model.eval()
    total_score = 0
    num_batches = 0
    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = modified_img.to(device), original_img.to(device)
            generated_img = model(modified_img)  

            score = LPIPS(generated_img, original_img, device) 

            total_score += score
            num_batches += 1
    average_score = total_score / num_batches
    print(f"Average LPIPS: {average_score:.4f}")
    return average_score


def LPIPS(original, generated, device):
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    device = next(lpips_loss_fn.parameters()).device
    original = original.unsqueeze(0).to(device)  # [1, C, H, W]
    generated = generated.unsqueeze(0).to(device)
    lpips_val = lpips_loss_fn(original, generated)
    return lpips_val.item()


