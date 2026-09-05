"""CLI entrypoint: python -m waterseg.cli --input in.tif --output out.tif"""

import argparse
import logging
import os

from .pipeline import run

# Configured once, here, in the actual entrypoint - not in pipeline.py, which
# just calls logging.getLogger(__name__) like any library module should.
# Timestamp + level on every line so batch-job output is filterable/parseable
# in CloudWatch (or any log aggregator), not just readable in a live terminal.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch waterbody segmentation inference (ONNX Runtime)")
    parser.add_argument("--input", required=True, help="Path to input GeoTIFF")
    parser.add_argument("--output", required=True, help="Path to write output mask GeoTIFF")
    args = parser.parse_args()

    # Fails fast with a clear message before touching rasterio/ONNX Runtime at
    # all - the most common real mistake (typo'd path, forgot to mount the data
    # volume) shouldn't need a traceback to diagnose in a batch job's logs.
    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    run(args.input, args.output)


if __name__ == "__main__":
    main()
