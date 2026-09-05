import itertools

import rasterio
from tqdm import tqdm

from .inference_onnx import predict_batch
from .model_onnx import get_providers, load_session
from .tiling import generate_tiles

TILE_SIZE = 512
OVERLAP = 256
BATCH_SIZE = 4
ONNX_MODEL_PATH = "model.onnx"  # baked into the image by Dockerfile.onnx's builder stage


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


def run(input_path: str, output_path: str) -> None:
    """Mirrors pipeline.py's run() exactly - same tiling/stitching loop, reusing
    tiling.generate_tiles unchanged. Only the model-loading and predict_batch
    calls differ (ONNX Runtime session instead of a torch.nn.Module).
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
