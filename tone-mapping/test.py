import os

import cv2
from numpy import ndarray

def read_exr(image_path: str):
    image = cv2.imread(
        image_path,
        flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def tone_map_reinhard(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapReinhard(
    gamma=2.2,
    intensity=0.0,
    light_adapt=0.0,
    color_adapt=0.0
    )
    result = tonemap_operator.process(src=image)
    return result
def tone_map_mantiuk(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapMantiuk(
    gamma=2.2,
    scale=0.85,
    saturation=1.2
    )
    result = tonemap_operator.process(src=image)
    return result

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
FILE_PATH = os.path.normpath("120.exr")

if __name__ == '__main__':
    image = read_exr(FILE_PATH)
    tone_mapped_reinhard = tone_map_reinhard(image)
    tone_mapped_mantiuk = tone_map_mantiuk(image)
    cv2.imshow("Original", image)
    cv2.imshow("Tone Mapped Reinhard", tone_mapped_reinhard)
    cv2.imshow("Tone Mapped Mantiuk", tone_mapped_mantiuk)
    cv2.waitKey(0)
    cv2.destroyAllWindows()