import numpy as np
import pyexr
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from PIL import Image


def display_height_plot(image):
    # Ensure the input is a 2D array
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Input to display_height_plot must be a 2D array.")

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    x, y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(x, y, image, cmap='hsv', rstride=1, cstride=1,  edgecolor='none')
    ax.plot_surface(x, y, image, cmap="hsv", edgecolor="none")
    ax.set_title("Brightness Plot")
    plt.show()


def display_pseudocolor(image, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap="hsv")
    plt.colorbar(label="Luminance")
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_exr(file_path, image):
    # Use pyexr.write to save the EXR file
    pyexr.write(file_path, image)


def save_png(file_path, image):
    # Ensure image values are in the range [0, 1]
    image_scaled = np.clip(image, 0, 1)
    # Convert to 8-bit and save as PNG
    image_uint8 = (image_scaled * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(file_path)


def save_pseudocolor_png(file_path, image):
    # Ensure image values are in the range [0, 1]
    if image.ndim != 2:
        raise ValueError(
            "Input image must be a 2D array for pseudocolor visualization."
        )
    image_scaled = np.clip(image / image.max(), 0, 1)

    # Apply the HSV colormap
    cmap = get_cmap("hsv")
    norm = Normalize(vmin=0, vmax=1)
    colored_image = cmap(norm(image_scaled))  # Apply colormap and normalize

    # Remove alpha channel and ensure proper shape and type
    rgb_image = (colored_image[:, :, :3] * 255).astype(
        np.uint8
    )  # Exclude alpha channel

    # Save as PNG
    Image.fromarray(rgb_image).save(file_path)
