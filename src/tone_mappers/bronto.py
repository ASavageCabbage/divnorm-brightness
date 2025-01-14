
import numpy as np
import cv2

def rgb_to_xyz(image):
    # Conversion matrix from sRGB to XYZdef xyz_to_lxy(X, Y, Z):
    rgb_to_xyz_matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                                  [0.2126729, 0.7151522, 0.0721750],
                                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = np.dot(image, rgb_to_xyz_matrix.T)
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]  # X, Y, Z


def xyz_to_lxy(X, Y, Z):
    denom = X + Y + Z + 1e-8  # Prevent division by zero
    x_chroma = X / denom
    y_chroma = Y / denom
    return Y, x_chroma, y_chroma  # Use Y as luminance


def lxy_to_rgb(luminance, x_chroma, y_chroma):
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
    xyz_to_rgb_matrix = np.array([[3.2404542, -1.5371385, -0.4985314],
                                  [-0.9692660,  1.8760108,  0.0415560],
                                  [0.0556434, -0.2040259,  1.0572252]])
    rgb = np.dot(np.stack([X, Y, Z], axis=-1), xyz_to_rgb_matrix.T)
    rgb = np.clip(rgb, 0, 1)  # Clip to valid range
    return rgb


def bronto(
    RGBimage: np.ndarray,
    sigma_0: float = 2.0,
    k: float = 0.85,
    N: int = 13):

    X, Y, Z = rgb_to_xyz(RGBimage)
    image, x_chroma, y_chroma = xyz_to_lxy(X, Y, Z)

    key = 0.25
    current_sigma = sigma_0
    power = 2.0
    non_linear_image = image**(1.0/power)
    current_k = 1
    center = cv2.GaussianBlur(non_linear_image, (0, 0), current_sigma)
    accum = np.zeros_like(image)
    response_sum = np.zeros_like(image)
    b = 1
    d = 1
    for i in range(N-1):
        s2 = current_sigma*current_sigma
        current_sigma *= 2
        surround = cv2.GaussianBlur(non_linear_image, (0, 0), current_sigma)
        response = current_k*(((center + b/s2) / (surround + d/s2)) - b/d )
        response = np.abs(response)
        accum += response*(center**power)
        response_sum  += response
        d = k*d
        current_k *= k
        center = surround
    local_white = accum / response_sum
    tonemapped_luminance = (key / local_white) * image
    return lxy_to_rgb(tonemapped_luminance, x_chroma, y_chroma)
