"""Smoke test: does the real container's plumbing survive, end to end?

Not a correctness check (that's check.py against the real ground-truth mask) -
this only asks "did anything break the basic contract": does the container run
against a tiny synthetic GeoTIFF without crashing, and does the output actually
match the input's width/height/CRS/transform, have exactly 1 band, uint8 dtype,
and contain only valid class values (0 or 1)? Fast (a few seconds, no real
Sentinel-2 data needed) so it can run after every change, not just before a
submission.

Runs the actual built image via `docker run` - the real deliverable, not a
Python-level shortcut - so requires `docker compose build` (or `docker build`)
to have been run first.

Usage:
    python tests/smoke_test.py
    python tests/smoke_test.py --image waterbody-segmentation:onnx
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

WIDTH = 600  # deliberately not a multiple of tile_size/stride (512/256) - forces
HEIGHT = 600  # edge-tile clamping in tiling.py to actually run, not just the common case
CHANNELS = 6
CRS = "EPSG:32636"  # same CRS family as the real Sentinel-2 sample data


def make_synthetic_input(path: Path) -> None:
    transform = from_origin(500000, 4649000, 10, 10)  # arbitrary origin, real 10m/px scale
    rng = np.random.default_rng(seed=0)
    data = rng.integers(0, 5000, size=(CHANNELS, HEIGHT, WIDTH), dtype="uint16")

    profile = {
        "driver": "GTiff",
        "width": WIDTH,
        "height": HEIGHT,
        "count": CHANNELS,
        "dtype": "uint16",
        "crs": CRS,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def run_container(image: str, data_dir: Path) -> None:
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "-v", f"{data_dir}:/data",
        image,
        "--input", "/data/synthetic_input.tif",
        "--output", "/data/synthetic_output.tif",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(f"Container exited with code {result.returncode}")


def check_output(input_path: Path, output_path: Path) -> None:
    with rasterio.open(input_path) as src:
        expected_width, expected_height = src.width, src.height
        expected_crs, expected_transform = src.crs, src.transform

    with rasterio.open(output_path) as dst:
        checks = [
            (dst.width == expected_width and dst.height == expected_height,
             f"shape mismatch: got {dst.width}x{dst.height}, expected {expected_width}x{expected_height}"),
            (dst.crs == expected_crs, f"CRS mismatch: got {dst.crs}, expected {expected_crs}"),
            (dst.transform == expected_transform,
             f"transform mismatch: got {dst.transform}, expected {expected_transform}"),
            (dst.count == 1, f"expected 1 band, got {dst.count}"),
            (dst.dtypes[0] == "uint8", f"expected uint8, got {dst.dtypes[0]}"),
        ]
        values = dst.read(1)
        checks.append((set(np.unique(values)) <= {0, 1}, f"non-binary values found: {np.unique(values)}"))

    failures = [msg for ok, msg in checks if not ok]
    if failures:
        raise SystemExit("FAILED:\n" + "\n".join(f"  - {m}" for m in failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="waterbody-segmentation:latest", help="Docker image to test")
    args = parser.parse_args()

    tmp_dir = Path(tempfile.mkdtemp(prefix="waterseg-smoke-"))
    try:
        input_path = tmp_dir / "synthetic_input.tif"
        output_path = tmp_dir / "synthetic_output.tif"

        make_synthetic_input(input_path)
        run_container(args.image, tmp_dir)
        check_output(input_path, output_path)

        print(f"PASSED: {args.image} produced a correctly-shaped, georeferenced, binary output mask.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
