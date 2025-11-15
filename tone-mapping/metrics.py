import numpy as np
import torch
from brisque import BRISQUE
from skimage.metrics import structural_similarity as ssim


def evaluate_brisque(image: torch.Tensor) -> float:
    metric = BRISQUE(url=False)

    img_conv = image.cpu().detach().numpy()
    img_conv = np.transpose(img_conv, (1, 2, 0))

    img_process = (img_conv * 255).astype(np.uint8)
    return metric.score(img=img_process)


# def evaluate_SSIM_batch(model: nn.Module, data: DataLoader, device: str):
#     model.eval()
#     total_score = 0
#     num_batches = 0
#     with torch.no_grad():
#         for modified_img, original_img in data:
#             modified_img, original_img = modified_img.to(device), original_img.to(device)
#             generated_img = model(modified_img)

#             score = SSIM(generated_img, original_img)

#             total_score += score.item()
#             num_batches += 1
#     average_score = total_score / num_batches
#     print(f"Average SSIM: {average_score:.4f}")
#     return average_score


def evaluate_SSIM(original, generated):
    score, dif = ssim(original, generated, full=True)
    return score
