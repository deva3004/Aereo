"""Build-time only: exports the pretrained model to ONNX. Not part of the runtime
inference path - runs once inside Dockerfile.onnx's builder stage, and the
resulting model.onnx is what ships in the final image.

Reuses model.load_model()/get_device() unchanged: this exports the exact same
in-memory model object the PyTorch pipeline loads and runs, so there is still
only one place in the codebase that constructs the model from the HF checkpoint.
"""

import torch

from .model import get_device, load_model

BATCH_SIZE = 4
TILE_SIZE = 512
CHANNELS = 6
OPSET_VERSION = 17


def export(output_path: str) -> None:
    device = get_device()
    model = load_model(device)

    dummy = torch.randn(BATCH_SIZE, CHANNELS, TILE_SIZE, TILE_SIZE, device=device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=OPSET_VERSION,
    )


if __name__ == "__main__":
    import sys

    export(sys.argv[1] if len(sys.argv) > 1 else "model.onnx")
