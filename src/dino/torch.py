import torch
from torchvision.transforms import GaussianBlur

from src.dino.util import *


def assert_no_nans(x: torch.Tensor):
    assert torch.isnan(x.view(-1)).sum().item() == 0


def rgb_to_xyz(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_to_xyz_matrix = torch.Tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    ).to(image.device)
    xyz = image @ rgb_to_xyz_matrix.T
    assert_no_nans(xyz)
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def gaussian_blur(
    image: torch.Tensor, sigma: int, kernel_sigmas: int = 1
) -> torch.Tensor:
    kernel_size = int(2 * kernel_sigmas * sigma + 1) | 1
    blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    image = blur(image).squeeze()
    assert_no_nans(image)
    return image


def dn_brightness_model(
    L: torch.Tensor,
    cs_ratio: float = 2.0,
    min_scale: float = 1.0,
    gamma: float = 2.2,
    w: float = 0.85,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    scale_normalized_constants: bool = False,
) -> torch.Tensor:
    L = scale_gamma(L, gamma=gamma)

    scales = generate_scales(L.size()[0], cs_ratio, min_scale)
    weights = [w**i for i in range(len(scales))]
    weighted_sum = torch.zeros(L.size()[:2]).to(L.device)
    center_response = gaussian_blur(L, scales[0])
    for i in range(1, len(scales)):
        surround_response = gaussian_blur(L, scales[i])
        assert center_response.ndim == 2
        assert surround_response.ndim == 2
        _b = b
        _d = d
        if scale_normalized_constants:
            _b /= scales[i] ** 2
            _d /= scales[i] ** 2
        weighted_sum += weights[i - 1] * (
            (a * center_response + _b) / (c * surround_response + _d) - _b / _d
        )
        center_response = surround_response
    assert_no_nans(weighted_sum)
    return weighted_sum
