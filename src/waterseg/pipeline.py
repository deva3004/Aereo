import itertools

import rasterio
from tqdm import tqdm

from .inference import predict_batch
from .model import get_device, load_model
from .tiling import generate_tiles

TILE_SIZE = 512
OVERLAP = 256
# Empirically tuned, not assumed — see learnings.md. Larger batches (8, 16) measured
# *slower* than this on CPU; this CPU has no meaningful throughput to gain from
# more parallelism, and larger batches likely just add memory-bandwidth/cache
# pressure instead. On real GPU hardware (the actual deployment target) the optimal
# value is likely different and higher, but unverified — no GPU available to test.
BATCH_SIZE = 4


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


def run(input_path: str, output_path: str) -> None:
    device = get_device()
    print(f"Using device: {device}")
    model = load_model(device)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)

        tiles = generate_tiles(src.width, src.height, TILE_SIZE, OVERLAP)

        with rasterio.open(output_path, "w", **profile) as dst:
            with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
                for batch in _batched(tiles, BATCH_SIZE):
                    imgs = [src.read(window=tile.read_window) for tile in batch]
                    cores = [tile.core for tile in batch]
                    preds = predict_batch(model, device, imgs, TILE_SIZE, cores)
                    for tile, pred_core in zip(batch, preds):
                        dst.write(pred_core[None, :, :], window=tile.write_window)
                    pbar.update(len(batch))
