import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def split_dataset(split_ration, dataset_path):
    train_ratio, val_ratio, _ = split_ration
    total_size = len(os.listdir(os.path.join(dataset_path, "images")))
    indices = list(range(total_size))

    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices


class PhongDataset(Dataset):
    def __init__(
        self,
        base_path,
        indices,
    ) -> None:
        self.image_path = os.path.join(base_path, "images")
        self.json_path = os.path.join(base_path, "dataset.json")
        self.transform = transforms.ToTensor()

        with open(self.json_path, "r") as f:
            all_samples = json.load(f)

        self.samples = [all_samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_vec = (
            sample["model_translation_relative"]
            + sample["material_diffuse"]
            + [sample["material_shininess"]]
            + sample["light_position_relative"]
        )
        input_tensor = torch.tensor(input_vec, dtype=torch.float32)

        image = Image.open(
            os.path.join(self.image_path, sample["image_filename"])
        ).convert("RGB")
        target_tensor = self.transform(image)

        return input_tensor, target_tensor
