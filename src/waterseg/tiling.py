"""Tile-grid geometry for sliding-window inference over a large raster.

Given a raster's width/height, produces the read window (with context margin)
and corresponding write window (the non-overlapping core) for every tile —
pure pixel-coordinate math, independent of any actual raster I/O or model.
"""

from dataclasses import dataclass

from rasterio.windows import Window


@dataclass(frozen=True)
class Tile:
    read_window: Window   # region to read from the source, includes context margin
    write_window: Window  # region in the output this tile's cropped prediction fills
    core: tuple[slice, slice]  # (row_slice, col_slice) into the read tile giving the core


def generate_tiles(
    width: int, height: int, tile_size: int = 512, overlap: int = 256
) -> list[Tile]:
    stride = tile_size - overlap
    margin = overlap // 2

    tiles = []
    for row_start in range(0, height, stride):
        core_row_end = min(row_start + stride, height) # cant go beyond the image height
        for col_start in range(0, width, stride):
            core_col_end = min(col_start + stride, width) # cant go beyond the image width

            read_row_start = max(row_start - margin, 0)
            read_col_start = max(col_start - margin, 0)
            read_row_end = min(core_row_end + margin, height)
            read_col_end = min(core_col_end + margin, width)

            core_row_offset = row_start - read_row_start
            core_col_offset = col_start - read_col_start
            core_height = core_row_end - row_start
            core_width = core_col_end - col_start

            tiles.append(
                Tile(
                    read_window=Window(
                        read_col_start,
                        read_row_start,
                        read_col_end - read_col_start,
                        read_row_end - read_row_start,
                    ),
                    write_window=Window(col_start, row_start, core_width, core_height),
                    core=(
                        slice(core_row_offset, core_row_offset + core_height),
                        slice(core_col_offset, core_col_offset + core_width),
                    ),
                )
            )
    return tiles
