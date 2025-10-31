import torch
import numpy as np
import opencv as cv2
from skimage.restoration import denoise_bilateral


def denoise_image(image: torch.Tensor) -> torch.Tensor:
    image_np = image.permute(1, 2, 0).np()
    denoised_image = denoise_bilateral(image_np)
    denoised_tensor = torch.from_numpy(denoised_image).float()
    if denoised_tensor.ndim == 3:
        denoised_tensor = denoised_tensor.permute(2, 0, 1)
    return denoised_tensor


def upscale_image(image: torch.Tensor, width: float, height: float) -> torch.Tensor:
    image_np = image.permute(1, 2, 0).np()
    new_size = (int(image_np.shape[1] * width), int(image_np.shape[0] * height))
    upscaled_image = cv2.resize(image_np, new_size, interpolation=cv2.INTER_CUBIC)
    upscaled_tensor = torch.from_numpy(upscaled_image).float()
    upscaled_tensor = upscaled_tensor.permute(2, 0, 1)
    return upscaled_tensor
