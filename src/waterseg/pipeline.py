import itertools
import os

import rasterio
from tqdm import tqdm

from .inference import predict_batch
from .model_onnx import get_providers, load_session
from .tiling import generate_tiles

TILE_SIZE = 512
OVERLAP = 256
# Empirically tuned - see learnings.md for the actual measured values (CPU and
# GPU). Overridable via env var so new hardware can be re-benchmarked without a
# rebuild, the same way the current default was chosen.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
ONNX_MODEL_PATH = "model.onnx"  # baked into the image by the Dockerfile's builder stage


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


def run(input_path: str, output_path: str) -> None:
    """End-to-end driver: opens input once, opens output once with corrected
    georeferencing, then streams tile-by-tile (read window -> predict -> write
    core) without ever holding the full image in memory - see tiling.py for
    the read/write window and overlap-crop geometry this loops over.
    """
    providers = get_providers()
    print(f"Using providers: {providers}")
    session = load_session(ONNX_MODEL_PATH, providers)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)

        tiles = generate_tiles(src.width, src.height, TILE_SIZE, OVERLAP)

        with rasterio.open(output_path, "w", **profile) as dst:
            with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
                for batch in _batched(tiles, BATCH_SIZE):
                    imgs = [src.read(window=tile.read_window) for tile in batch]
                    cores = [tile.core for tile in batch]
                    preds = predict_batch(session, imgs, TILE_SIZE, cores)
                    for tile, pred_core in zip(batch, preds):
                        dst.write(pred_core[None, :, :], window=tile.write_window)
                    pbar.update(len(batch))
