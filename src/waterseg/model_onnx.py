import onnxruntime as ort

# Mirrors model.py's get_device()/load_model() shape on purpose - same two
# responsibilities (pick a device/provider, load a ready-to-run model object),
# different backend.


def get_providers() -> list[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_session(onnx_path: str, providers: list[str]) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, providers=providers)
