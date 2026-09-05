import itertools
import os

import rasterio
from tqdm import tqdm

from .inference import predict_batch
from .model_onnx import get_providers, load_session
from .tiling import generate_tiles

TILE_SIZE = 512
OVERLAP = 256
# Empirically tuned on real GPU hardware - see learnings.md. 8 measured ~1.3%
# faster than 4 on the actual deployment target (116.1s vs 117.6s full-image,
# identical accuracy) - a much smaller gap than the CPU-only tuning found (4 was
# ~65% faster than 8 there), suggesting this GPU already saturates around batch
# size 4 and the remaining bottleneck is elsewhere (I/O/preprocessing, not
# compute). Overridable via env var so different hardware can be re-benchmarked
# without a rebuild.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
ONNX_MODEL_PATH = "model.onnx"  # baked into the image by the Dockerfile's builder stage
# model.onnx was exported with a fixed 6-channel input shape (see export_onnx.py's
# CHANNELS) - Sentinel-2 B2/B3/B4/B8/B11/B12. A different band count would
# otherwise fail deep inside session.run(), tile by tile, with a much less
# readable ONNX Runtime shape-mismatch error.
EXPECTED_CHANNELS = 6


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

    try:
        src_context = rasterio.open(input_path)
    except rasterio.errors.RasterioIOError as e:
        raise SystemExit(f"Could not open input as a raster: {input_path} ({e})")

    with src_context as src:
        if src.count != EXPECTED_CHANNELS:
            raise SystemExit(
                f"Expected {EXPECTED_CHANNELS} bands (Sentinel-2 B2,B3,B4,B8,B11,B12), "
                f"got {src.count} in {input_path}"
            )

        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)

        tiles = generate_tiles(src.width, src.height, TILE_SIZE, OVERLAP)

        try:
            dst_context = rasterio.open(output_path, "w", **profile)
        except rasterio.errors.RasterioIOError as e:
            raise SystemExit(f"Could not open output path for writing: {output_path} ({e})")

        with dst_context as dst:
            with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
                for batch in _batched(tiles, BATCH_SIZE):
                    imgs = [src.read(window=tile.read_window) for tile in batch]
                    cores = [tile.core for tile in batch]
                    preds = predict_batch(session, imgs, TILE_SIZE, cores)
                    for tile, pred_core in zip(batch, preds):
                        dst.write(pred_core[None, :, :], window=tile.write_window)
                    pbar.update(len(batch))
