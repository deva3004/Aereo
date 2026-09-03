import numpy as np
import torch

NORMALIZE_SCALE = 255.0


def predict_batch(
    model: torch.nn.Module,
    device: torch.device,
    imgs: list[np.ndarray],
    tile_size: int,
    cores: list[tuple[slice, slice]],
) -> list[np.ndarray]:
    """imgs: each (C, h, w) array read from a source raster window (h, w <= tile_size).

    Batches tiles into one forward pass for throughput. Numerically identical to
    predicting each tile alone: model.eval() (set in model.py) makes BatchNorm use
    fixed running statistics rather than per-batch statistics, so one tile's
    prediction doesn't depend on what else is in its batch.
    """
    padded = []
    for img in imgs:
        channels, h, w = img.shape
        img = img.astype(np.float32) / NORMALIZE_SCALE
        if h < tile_size or w < tile_size:
            pad = np.zeros((channels, tile_size, tile_size), dtype=np.float32)
            pad[:, :h, :w] = img
            img = pad
        padded.append(img)

    batch = np.stack(padded, axis=0)
    tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

    return [preds[i][row, col] for i, (row, col) in enumerate(cores)]
