import os
from typing import Any, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root: str, noise_level: float = None, input_size: Tuple[int, int] = None):
        self.root = root
        self.image_paths = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".png")
        ]
        self.transform = transforms.ToTensor()
        if noise_level and input_size:
            raise ValueError("Only one value can be set at the same time.")
        self.noise_level = noise_level
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.__resize(img, (256, 256))
        if self.input_size is not None:
            modified_img = self.__resize(img, self.input_size, interpolation=cv2.INTER_AREA)
            modified_img = modified_img.astype(np.float32) / 255.0
            modified_img = self.transform(modified_img)

        img = img.astype(np.float32) / 255.0
        img = self.transform(img)

        if self.noise_level is not None:
            modified_img = self.__add_noise(img, self.noise_level)

        return modified_img, img
    
    def __add_noise(self, img: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        noise = torch.randn_like(img) * noise_level
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        return noisy_img

    def __resize(self, img: Any, size: Tuple[int, int], interpolation: int = None) -> Any:
        img = cv2.resize(img, size, interpolation=interpolation or cv2.INTER_LINEAR)
        return img