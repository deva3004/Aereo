import numpy as np
import onnxruntime as ort

NORMALIZE_SCALE = 255.0


def predict_batch(
    session: ort.InferenceSession,
    imgs: list[np.ndarray],
    tile_size: int,
    cores: list[tuple[slice, slice]],
) -> list[np.ndarray]:
    """Normalizes each tile, zero-pads edge tiles up to tile_size, runs one
    ONNX Runtime session.run() over the whole batch, and crops each result
    back down to its tile's core region (see tiling.py for what 'core' means).
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
    logits = session.run(["output"], {"input": batch})[0]
    preds = np.argmax(logits, axis=1).astype(np.uint8)

    return [preds[i][row, col] for i, (row, col) in enumerate(cores)]
