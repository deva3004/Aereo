"""CLI entrypoint: python -m waterseg.cli --input in.tif --output out.tif"""

import argparse
import os

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch waterbody segmentation inference (ONNX Runtime)")
    parser.add_argument("--input", required=True, help="Path to input GeoTIFF")
    parser.add_argument("--output", required=True, help="Path to write output mask GeoTIFF")
    args = parser.parse_args()

    # Fails fast with a clear message before touching rasterio/ONNX Runtime at
    # all - the most common real mistake (typo'd path, forgot to mount the data
    # volume) shouldn't need a traceback to diagnose in a batch job's logs.
    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    run(args.input, args.output)


if __name__ == "__main__":
    main()
