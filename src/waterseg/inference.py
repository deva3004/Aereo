"""Runs the model on a single tile and returns its cropped core prediction."""

import numpy as np
import torch

# Empirically validated against expected_mask.tif — see problems.md.
NORMALIZE_SCALE = 255.0


def predict_tile(
    model: torch.nn.Module,
    device: torch.device,
    img: np.ndarray,
    tile_size: int,
    core: tuple[slice, slice],
) -> np.ndarray:
    """img: (C, h, w) array read from the source raster window (h, w <= tile_size)."""
    channels, h, w = img.shape
    img = img.astype(np.float32) / NORMALIZE_SCALE

    if h < tile_size or w < tile_size:
        padded = np.zeros((channels, tile_size, tile_size), dtype=np.float32)
        padded[:, :h, :w] = img
        img = padded

    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    row_slice, col_slice = core
    return pred[row_slice, col_slice]
