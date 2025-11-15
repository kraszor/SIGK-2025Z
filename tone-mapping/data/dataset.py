import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import Dataset
from utils import generate_exposures, normalize_hdr_image, read_exr_file


class ToneMappingDataset(Dataset):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.files = [f for f in os.listdir(dir_path) if f.endswith(".exr")]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        base_filename = self.files[idx]
        file_path = os.path.join(self.dir_path, base_filename)

        hdr_image = read_exr_file(file_path)
        hdr_image = normalize_hdr_image(hdr_image)
        e_low, e_mid, e_high = generate_exposures(hdr_image)

        low_exposure = torch.tensor(e_low, dtype=torch.float32).permute(2, 0, 1)
        mid_exposure = torch.tensor(e_mid, dtype=torch.float32).permute(2, 0, 1)
        high_exposure = torch.tensor(e_high, dtype=torch.float32).permute(2, 0, 1)
        hdr_image = torch.tensor(hdr_image, dtype=torch.float32).permute(2, 0, 1)

        return low_exposure, mid_exposure, high_exposure, hdr_image


# def tone_map_reinhard(image: ndarray) -> ndarray:
#     tonemap_operator = cv2.createTonemapReinhard(
#     gamma=2.2,
#     intensity=0.0,
#     light_adapt=0.0,
#     color_adapt=0.0
#     )
#     result = tonemap_operator.process(src=image)
#     return result
# def tone_map_mantiuk(image: ndarray) -> ndarray:
#     tonemap_operator = cv2.createTonemapMantiuk(
#     gamma=2.2,
#     scale=0.85,
#     saturation=1.2
#     )
#     result = tonemap_operator.process(src=image)
#     return result
# def evaluate_image(image: ndarray) -> float:
#     metric = BRISQUE(url=False)
#     return metric.score(img=image)
# if __name__ == '__main__':
#     image = read_exr(im_path=FILE_PATH)
#     tone_mapped_reinhard = tone_map_reinhard(image)
#     tone_mapped_mantiuk = tone_map_mantiuk(image)
#     cv2.imshow('original', image)
#     cv2.imshow('tone_mapped_reinhard', tone_mapped_reinhard)
#     cv2.imshow('tone_mapped_mantiuk', tone_mapped_mantiuk)
#     print('tone_mapped_reinhard', evaluate_image(image=tone_mapped_reinhard))
#     print('tone_mapped_mantiuk', evaluate_image(image=tone_mapped_mantiuk))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
