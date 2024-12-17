import cv2
import numpy as np
from PIL import Image


def read_image(
    file_path: str,
    white_nits: float = 400,
    resize_width: int = None,
    resize_height: int = None,
) -> np.ndarray:
    """Load HDR or PNG image from disk as RGB numpy array.

    Assumes EXR images are in units of nits (candela/m^2).
    Assumes PNG images are displayed on a monitor at the specified luminance for pure white.
    """
    if any(file_path.endswith(ext) for ext in [".exr", ".hdr"]):
        gamma = 2.2
        hdr_image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        # Check that image is RGB
        assert hdr_image.shape[2] == 3
        # OpenCV uses BGR
        b = hdr_image[:, :, 0]
        g = hdr_image[:, :, 1]
        r = hdr_image[:, :, 2]
        image = np.stack([r, g, b], axis=-1)
        r = np.absolute(r)
        g = np.absolute(g)
        b = np.absolute(b)
        r = np.power(r, 1.0 / gamma)
        g = np.power(g, 1.0 / gamma)
        b = np.power(b, 1.0 / gamma)
    elif file_path.endswith(".png"):
        image = white_nits * (
            np.asarray(Image.open(file_path).convert("RGB"), dtype=np.float32) / 255.0
        )
    else:
        raise ValueError(
            "Unsupported file format. Only HDR/EXR and PNG files are supported."
        )
    # Optionally resize the image before output
    (
        height,
        width,
    ) = image.shape[:2]
    if not (resize_width is None and resize_height is None):
        if resize_width is None:
            resize_width = resize_height / height * width
        if resize_height is None:
            resize_height = resize_width / width * height
        resize_width = round(resize_width)
        resize_height = round(resize_height)
        if height != resize_height or width != resize_width:
            image = cv2.resize(
                image, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC
            )
    return image.astype(np.float32)


def rgb_to_xyz(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Conversion matrix from sRGB to XYZ
    rgb_to_xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = np.dot(image, rgb_to_xyz_matrix.T)
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]  # X, Y, Z


def xyz_to_lxy(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denom = X + Y + Z + 1e-8  # Prevent division by zero
    x_chroma = X / denom
    y_chroma = Y / denom
    return Y, x_chroma, y_chroma  # Use Y as luminance


def lxy_to_rgb(
    luminance: np.ndarray, x_chroma: np.ndarray, y_chroma: np.ndarray
) -> np.ndarray:
    # Ensure all inputs are 2D arrays
    luminance = np.squeeze(luminance)
    x_chroma = np.squeeze(x_chroma)
    y_chroma = np.squeeze(y_chroma)

    # Validate shapes
    if luminance.shape != x_chroma.shape or luminance.shape != y_chroma.shape:
        raise ValueError(
            f"Shape mismatch: luminance {luminance.shape}, x_chroma {x_chroma.shape}, y_chroma {y_chroma.shape}"
        )

    z_chroma = 1 - x_chroma - y_chroma
    X = luminance * x_chroma / (y_chroma + 1e-8)
    Z = luminance * z_chroma / (y_chroma + 1e-8)
    Y = luminance
    xyz_to_rgb_matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb = np.dot(np.stack([X, Y, Z], axis=-1), xyz_to_rgb_matrix.T)
    rgb = np.clip(rgb, 0, 1)  # Clip to valid range
    return rgb


def gaussian_blur_cv(image: np.ndarray, sigma: float) -> np.ndarray:
    # Use OpenCV's GaussianBlur
    kernel_size = int(6 * sigma + 1) | 1  # Ensure the kernel size is odd
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )


def dn_brightness_model(
    L: np.ndarray,
    min_scale: int = 1,
    w: float = 0.85,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    scale_normalized_constants: bool = False,
) -> np.ndarray:
    """Apply divisive normalization brightness model to array of luminances (i.e. greyscale input image).

    B(x,y) = sum(
        w**i * (
            (a*center + b / scales[i]**2) / (c*surround + d / scales[i]**2)
            - (b / scales[i]**2) / (d / scales[i]**2)
        )
    )

    NOTE: The current scale is the center stdev at the current scale,
          the next scale up is the surround stdev at the current scale.
    """
    max_scale = np.min(L.shape)
    scale = max_scale
    scales = []
    while scale / 2 >= min_scale:
        next_scale = scale / 2
        scales.insert(0, next_scale)
        scale = next_scale
    weights = [w**i for i in range(len(scales))]

    # Initialize weighted_sum as a 2D array
    weighted_sum = np.zeros(L.shape[:2], dtype=L.dtype)

    # Compute ratios and weighted sum using only two blurred images at a time
    center_response = gaussian_blur_cv(L, scales[0])
    center_response = np.squeeze(center_response)  # Ensure 2D

    for i in range(1, len(scales)):
        surround_response = gaussian_blur_cv(L, scales[i])
        surround_response = np.squeeze(surround_response)  # Ensure 2D

        # Ensure ratio computation works with 2D arrays
        if center_response.ndim != 2 or surround_response.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays, got shapes {center_response.shape}, {surround_response.shape}"
            )

        _b = b
        _d = d
        if scale_normalized_constants:
            _b /= scales[i] ** 2
            _d /= scales[i] ** 2

        weighted_sum += weights[i - 1] * (
            (a * center_response + _b) / (c * surround_response + _d) - _b / _d
        )
        center_response = surround_response  # Update for the next iteration

    return weighted_sum
