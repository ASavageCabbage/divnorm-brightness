import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def image_height_plot(
    image: np.ndarray,
    title: str = "",
    colorscale: str = "Viridis",
    output_path: str = "",
    display: bool = True,
):
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
            aspectmode="manual",
            aspectratio=dict(
                x=1,
                y=len(y) / len(x),
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


def image_pseudocolor_plot(
    image: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    output_path: str = "",
    display: bool = False,
    label: str = "brightness response",
):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(image, cmap=cmap, aspect="equal")
    cax = ax.inset_axes((1.05, 0, 0.08, 1.0))
    fig.colorbar(im, cax=cax, label=label)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if display:
        plt.show()
