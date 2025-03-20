from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize
from utils.randaugment import RandomAugment
import torch
import torchvision.transforms.functional as FT
import math
import torch.nn.functional as F

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)

class transform_train:
    def __init__(self, image_size=384, min_scale=0.5):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, img):
        return self.transform(img)


# class transform_train:
#     def __init__(self, image_size=384, min_scale=0.5):
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     image_size,
#                     scale=(min_scale, 1.0),
#                     interpolation=InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomHorizontalFlip(),
#                 RandomAugment(
#                     2,
#                     5,
#                     isPIL=True,
#                     augs=[
#                         "Identity",
#                         "AutoContrast",
#                         "Brightness",
#                         "Sharpness",
#                         "Equalize",
#                         "ShearX",
#                         "ShearY",
#                         "TranslateX",
#                         "TranslateY",
#                         "Rotate",
#                     ],
#                 ),
#                 transforms.ToTensor(),
#                 normalize,
#             ]
#         )

#     def __call__(self, img):
#         return self.transform(img)


class transform_test(transforms.Compose):
    def __init__(self, image_size=384):
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, img):
        return self.transform(img)



class TargetPad:
    """
    If an image aspect ratio is above a target ratio, pad the image to match such target ratio.
    For more details see Baldrati et al. 'Effective conditioned and composed image retrieval combining clip-based features.' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022).
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return FT.pad(image, padding, 0, 'constant')


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def targetpad_transform(target_ratio: float, dim: int) -> torch.Tensor:
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])