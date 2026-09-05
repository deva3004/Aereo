import ctypes

import onnxruntime as ort

# Mirrors model.py's get_device()/load_model() shape on purpose - same two
# responsibilities (pick a device/provider, load a ready-to-run model object),
# different backend.


def _cuda_driver_present() -> bool:
    # ort.get_available_providers() only means CUDA support was compiled into
    # this onnxruntime-gpu build - not that a real GPU/driver exists right now.
    # Confirmed the hard way: requesting CUDAExecutionProvider when the pip-
    # bundled math libraries (cublas etc.) are present but the real NVIDIA
    # driver stub (libcuda.so.1, only ever injected by nvidia-container-toolkit
    # under `docker run --gpus all`) is absent segfaults inside onnxruntime's
    # C++ session creation - not a catchable Python exception. Checking for the
    # driver library ourselves first, the same way torch.cuda.is_available()
    # safely probes for a GPU, avoids ever reaching that crash path.
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def get_providers() -> list[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available and _cuda_driver_present():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_session(onnx_path: str, providers: list[str]) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, providers=providers)
