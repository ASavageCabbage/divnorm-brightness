"""Example for how to run DINOS and BRONTO"""

import os

import matplotlib.pyplot as plt

import bronto


# Needed to use OpenEXR with OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# Brightness response demonstration

IMAGE_FILE = "noisy_contrast.png"
WHITE_NITS = 200  # assuming display device has 200 nits at pure white

image = bronto.read_image(IMAGE_FILE)
L = bronto.rgb_to_relative_luminance(image) * WHITE_NITS
brightness_response = bronto.dinos_model(L)
print("Calculated brightness response for", IMAGE_FILE)

fig, ax = plt.subplots(1, 1)
im = ax.imshow(brightness_response, cmap="viridis")
fig.colorbar(im)
fig.tight_layout()
fig.savefig("brightness.png", dpi=1200)
print("Exported heatmap of brightness response to brightness.png")


# BRONTO tone mapper demonstration

EXR_FILE = "507.exr"

image = bronto.read_image(EXR_FILE)
# resize it smaller so it doesn't take forever to tone map
image = bronto.downsize_image(image, resize_width=1600)
tonemapped_image = bronto.bronto(image)
bronto.write_image("bronto_tonemapped.png", tonemapped_image)
print("Exported tonemapped LDR image for", EXR_FILE)
