"""End-to-end batch inference: read input.tif, predict per tile, write output.tif."""

import rasterio
from tqdm import tqdm

from .inference import predict_tile
from .model import get_device, load_model
from .tiling import generate_tiles

TILE_SIZE = 512
OVERLAP = 256


def run(input_path: str, output_path: str) -> None:
    device = get_device()
    print(f"Using device: {device}")
    model = load_model(device)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)

        tiles = generate_tiles(src.width, src.height, TILE_SIZE, OVERLAP)

        with rasterio.open(output_path, "w", **profile) as dst:
            for tile in tqdm(tiles, desc="Processing tiles"):
                img = src.read(window=tile.read_window)
                pred_core = predict_tile(model, device, img, TILE_SIZE, tile.core)
                dst.write(pred_core[None, :, :], window=tile.write_window)
