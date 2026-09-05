"""Compares a predicted mask against the ground-truth mask, pixel-for-pixel.

This is the actual correctness bar the project runs against every change - a
real batching bug once produced a plausible-looking output that "ran without
error" but was silently wrong (see the README's "Findings & errors"). A clean
run is not evidence of a correct one; this script is.

Usage:
    python check.py                                   # data/output.tif vs data/expected_mask.tif
    python check.py --pred data/output-gpu-b8.tif      # compare a different prediction
    python check.py --pred a.tif --expected b.tif      # compare any two masks
"""

import argparse

import numpy as np
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(description="Pixel-accuracy/IoU check against a ground-truth mask")
    parser.add_argument("--pred", default="data/output.tif", help="Path to the predicted mask GeoTIFF")
    parser.add_argument("--expected", default="data/expected_mask.tif", help="Path to the ground-truth mask GeoTIFF")
    args = parser.parse_args()

    with rasterio.open(args.pred) as src:
        pred = src.read(1)
    with rasterio.open(args.expected) as src:
        expected = src.read(1)

    if pred.shape != expected.shape:
        raise SystemExit(f"Shape mismatch: pred {pred.shape} vs expected {expected.shape}")

    pred = (pred > 0).astype(np.uint8)
    expected = (expected > 0).astype(np.uint8)

    correct = (pred == expected).sum()
    total = pred.size
    accuracy = correct / total

    tp = np.logical_and(pred == 1, expected == 1).sum()
    fp = np.logical_and(pred == 1, expected == 0).sum()
    fn = np.logical_and(pred == 0, expected == 1).sum()
    iou = tp / (tp + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    pred_water_frac = pred.sum() / pred.size
    expected_water_frac = expected.sum() / expected.size

    print(f"Pixel accuracy: {accuracy:.5f}")
    print(f"IoU: {iou:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"Predicted water fraction: {pred_water_frac:.4f}  (ground truth: {expected_water_frac:.4f})")


if __name__ == "__main__":
    main()
