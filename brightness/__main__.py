import argparse

from brightness import brightness


def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI for applying Blommaert and Martens's "
        "object-oriented brightness perception model on images."
    )
    parser.add_argument("--image-path", "-i", required=True)
    parser.add_argument("--output-path", "-o", default="")
    parser.add_argument("--cmap", default=brightness.DEFAULT_CMAP)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=brightness.DEFAULT_SCALES
    )
    parser.add_argument(
        "--ratio", type=float, default=brightness.DEFAULT_CENTER_SURROUND_RATIO
    )
    parser.add_argument(
        "--flux", type=float, default=brightness.DEFAULT_TRANSITION_FLUX
    )
    parser.add_argument(
        "--decay", type=float, default=brightness.DEFAULT_EXPONENTIAL_WEIGHT_DECAY
    )
    return parser.parse_args()


if __name__ == "__main__":
    parser = get_parser()
    brightness.plot_brightness(
        parser.image_path,
        output_path=parser.output_path,
        cmap=parser.cmap,
        scales=parser.scales,
        center_surround_ratio=parser.ratio,
        transition_flux=parser.flux,
        exponential_weight_decay=parser.decay,
    )
