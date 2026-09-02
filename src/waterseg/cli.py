"""CLI entrypoint: python -m waterseg.cli --input in.tif --output out.tif"""

import argparse

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch waterbody segmentation inference")
    parser.add_argument("--input", required=True, help="Path to input GeoTIFF")
    parser.add_argument("--output", required=True, help="Path to write output mask GeoTIFF")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
