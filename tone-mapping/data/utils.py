import cv2
import numpy as np
import torch

def read_exr_file(image_path: str):
        image = cv2.imread(
            image_path,
            flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def normalize_hdr_image(hdr_image):
    mean_intensity = hdr_image.mean()
    normalized_image = (0.5 * hdr_image) / mean_intensity
    return normalized_image


def generate_exposures(hdr_image):
    x_p = 1.21497
    log2 = torch.log(torch.tensor(2.0))

    c_start = torch.log(x_p / hdr_image.max()) / log2
    c_end = torch.log(x_p / torch.quantile(hdr_image, 0.5)) / log2

    c_mid = (c_start + c_end) / 2

    e_low = np.clip((2 ** c_start) * hdr_image, 0, 1)
    e_mid = np.clip((2 ** c_mid) * hdr_image, 0, 1)
    e_high = np.clip((2 ** c_end) * hdr_image, 0, 1)

    return e_low, e_mid, e_high
