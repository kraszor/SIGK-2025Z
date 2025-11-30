import cv2
import flip_evaluator as flip
import lpips
import torch
import torch.nn as nn
from skimage.metrics import hausdorff_distance
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader


def LPIPS(original, generated, device):
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
    device = next(lpips_loss_fn.parameters()).device
    original = original.unsqueeze(0).to(device)
    generated = generated.unsqueeze(0).to(device)
    lpips_val = lpips_loss_fn(original, generated)
    return lpips_val.item()


def SSIM(original, generated):
    score, _ = ssim(original, generated, full=True)
    return score


def FLIP(original, generated):
    _, mean_flip_error, _ = flip.evaluate(original, generated, "LDR")
    return mean_flip_error


def hausdorff_distance_metric(original, generated):
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(generated):
        generated = generated.cpu().numpy()

    if len(original.shape) == 3 and original.shape[0] in [1, 3, 4]:
        original = original.transpose(1, 2, 0)
    if len(generated.shape) == 3 and generated.shape[0] in [1, 3, 4]:
        generated = generated.transpose(1, 2, 0)

    if original.max() <= 1.0:
        original = (original * 255).astype("uint8")
    else:
        original = original.astype("uint8")

    if generated.max() <= 1.0:
        generated = (generated * 255).astype("uint8")
    else:
        generated = generated.astype("uint8")

    if len(original.shape) == 3 and original.shape[2] == 3:
        original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    elif len(original.shape) == 3 and original.shape[2] == 1:
        original = original.squeeze(2)
    
    if len(generated.shape) == 3 and generated.shape[2] == 3:
        generated = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    elif len(generated.shape) == 3 and generated.shape[2] == 1:
        generated = generated.squeeze(2)
    t_lower = 50
    t_upper = 150
    edge_original = cv2.Canny(original, t_lower, t_upper)
    cv2.imshow("Edge Original", edge_original)
    cv2.waitKey(1)
    edge_generated = cv2.Canny(generated, t_lower, t_upper)
    cv2.imshow("Edge Generated", edge_generated)
    cv2.waitKey(1)
    return hausdorff_distance(edge_original, edge_generated)


def evaluate_SSIM(model: nn.Module, data: DataLoader, device: str):
    model.eval()
    total_score = 0
    num_batches = 0
    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = (
                modified_img.to(device),
                original_img.to(device),
            )
            generated_img = model(modified_img)
            score = SSIM(original_img, generated_img)
            total_score += score.item()
            num_batches += 1
    average_score = total_score / num_batches
    print(f"Average SSIM: {average_score:.4f}")
    return average_score


def evaluate_LPIPS(model: nn.Module, data: DataLoader, device: str):
    model.eval()
    total_score = 0
    num_batches = 0
    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = (
                modified_img.to(device),
                original_img.to(device),
            )
            generated_img = model(modified_img)
            score = LPIPS(original_img, generated_img, device)
            total_score += score
            num_batches += 1
    average_score = total_score / num_batches
    print(f"Average LPIPS: {average_score:.4f}")
    return average_score


def evaluete_FLIP(model: nn.Module, data: DataLoader, device: str):
    model.eval()
    total_score = 0
    num_batches = 0
    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = (
                modified_img.to(device),
                original_img.to(device),
            )
            generated_img = model(modified_img)
            score = FLIP(original_img, generated_img)
            total_score += score
            num_batches += 1
    average_score = total_score / num_batches
    print(f"Average FLIP: {average_score:.4f}")
    return average_score


def evaluate_hausdorff_distance(model: nn.Module, data: DataLoader, device: str):
    model.eval()
    total_distance = 0
    num_batches = 0
    with torch.no_grad():
        for modified_img, original_img in data:
            modified_img, original_img = (
                modified_img.to(device),
                original_img.to(device),
            )
            generated_img = model(modified_img)
            distance = hausdorff_distance_metric(original_img, generated_img)
            total_distance += distance
            num_batches += 1
    average_distance = total_distance / num_batches
    print(f"Average Hausdorff Distance: {average_distance:.4f}")
    return average_distance
