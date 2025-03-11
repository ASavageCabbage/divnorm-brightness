from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

from utils import downsample_image, generate_scales, read_image


def _assert_no_nans(x: torch.Tensor):
    assert torch.isnan(x.view(-1)).sum().item() == 0


def _gaussian_blur(
    image: torch.Tensor, sigma: int, kernel_sigmas: int = 2
) -> torch.Tensor:
    kernel_size = int(2 * kernel_sigmas * sigma + 1) | 1
    kernel_size = min(kernel_size, min(*image.size()) * 2 - 1)
    blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    if len(image.size()) < 3:
        image = image.unsqueeze(0)
    image = blur(image).squeeze()
    _assert_no_nans(image)
    return image


def _dinos_pytorch(
    L: torch.Tensor,
    gamma: float = 2.2,
    cs_ratio: float = 2.0,
    num_scales: int = 13,
    w: float = 0.9,
    b: float = 1.0,
    d: float = 1.0,
) -> torch.Tensor:
    L = L ** (1.0 / gamma)

    scales = generate_scales(L.size()[1], cs_ratio, num_scales)
    weights = [w**i for i in range(len(scales))]
    weighted_sum = torch.zeros(L.size()[:2]).to(L.device)
    center_response = _gaussian_blur(L, scales[0])
    for i in range(1, len(scales)):
        surround_response = _gaussian_blur(L, scales[i])
        assert center_response.ndim == 2
        assert surround_response.ndim == 2
        _b = b / scales[i] ** 2
        _d = b / scales[i] ** 2
        weighted_sum += weights[i - 1] * (
            (center_response + _b) / (surround_response + _d) - _b / _d
        )
        center_response = surround_response
    _assert_no_nans(weighted_sum)
    return weighted_sum


def _fairchild_to_relative_luminance(image: np.ndarray) -> np.ndarray:
    return 0.1904 * image[:, :, 0] + 0.7646 * image[:, :, 1] + 0.0450 * image[:, :, 2]


def _clip_and_get_brightness(image, white_nits):
    # Clip results to (0, 1)
    zeros = torch.zeros(image.size()).to(image.device)
    image = torch.maximum(image, zeros)
    ones = torch.full(image.size(), 1).to(image.device)
    image = torch.minimum(image, ones)
    # Compute brightness response
    L = _fairchild_to_relative_luminance(image) * white_nits
    brightness_response = _dinos_pytorch(L)
    return brightness_response.unsqueeze(0)


class ACESModel(nn.Module):
    def __init__(
        self, key=0.18, a=2.51, b=0.03, c=2.43, d=0.59, e=0.14, white_nits=200
    ):
        super(ACESModel, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([key, a, b, c, d, e]))
        self.white_nits = white_nits

    def forward(self, x):
        key, a, b, c, d, e = self.weights
        x = (key / (torch.mean(x**2) ** (1 / 2))) * x
        tonemapped_x = torch.squeeze((x * (a * x + b)) / (x * (c * x + d) + e))
        return _clip_and_get_brightness(tonemapped_x, self.white_nits)


class ReinhardModel(nn.Module):
    def __init__(self, key=0.18, l_white=0.5, gamma=2.2, white_nits=200):
        super(ReinhardModel, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([key, l_white]))
        self.gamma = gamma
        self.white_nits = white_nits

    def forward(self, x):
        key, l_white = self.weights
        x = (key / (torch.mean(x ** (1.0 / self.gamma))) ** self.gamma) * x
        tonemapped_x = torch.squeeze(x / (1 + x / (l_white**2)))
        return _clip_and_get_brightness(tonemapped_x, self.white_nits)


class BrightnessDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, downsample_ratio: float = 1.0):
        self.input_file_paths = sorted(
            f for f in Path(input_dir).iterdir() if f.is_file()
        )
        self.target_file_paths = sorted(
            f for f in Path(target_dir).iterdir() if f.is_file()
        )
        self.downsample_ratio = downsample_ratio

        assert all(
            self.input_file_paths[i].stem == self.target_file_paths[i].stem
            for i in range(len(self.input_file_paths))
        )

    def __len__(self):
        return len(self.input_file_paths)

    def __getitem__(self, idx):
        input_image = read_image(str(self.input_file_paths[idx]), min_value=1e-8)
        input_image = downsample_image(
            input_image,
            resize_width=round(input_image.shape[1] / self.downsample_ratio),
        )
        brightness_response = np.load(self.target_file_paths[idx])
        return input_image, brightness_response
