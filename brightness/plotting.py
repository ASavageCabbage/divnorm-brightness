import numpy as np
import pyexr
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from PIL import Image


def image_height_plot(
    image: np.ndarray,
    title: str = "",
    colorscale: str = "Viridis",
    output_path: str = "",
    display: bool = True,
) -> go.Figure:
    # Ensure the input is a 2D array
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Input to display_height_plot must be a 2D array.")

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    # Invert the y data to display the image properly
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])[::-1]
    X, Y = np.meshgrid(x, y)
    fig = go.Figure(
        data=[
            go.Surface(
                z=image,
                x=X,
                y=Y,
                hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z:.2f}",
                colorscale=colorscale,
            )
        ],
    )
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(
                x=1,
                y=len(y)/len(x),
                z=0.5,
            ),
        ),
    )
    if title:
        fig.update_layout(title=title)
    if output_path:
        fig.write_html(output_path)
    if display:
        fig.show()
    return fig


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
