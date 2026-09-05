
# Build-time only: used exclusively by export_onnx.py, inside the Dockerfile's
# builder stage, to construct the pretrained architecture and load its weights
# so it can be traced to ONNX. Never imported at actual inference time - the
# runtime path loads the exported model.onnx via model_onnx.py's ONNX Runtime
# session instead, which is why this file and model_onnx.py both exist.
import json

import segmentation_models_pytorch as smp # build the Unet++ model with EfficientNet-B4 encoder
import torch # used for loading the model and running inference on the GPU or CPU
from huggingface_hub import hf_hub_download

REPO_ID = "giswqs/s2-water-unetplusplus-efficientnet-b4"

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device) -> torch.nn.Module:
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")

    with open(config_path) as f:
        config = json.load(f)

    model = smp.create_model(
        arch=config["architecture"],
        encoder_name=config["encoder_name"],
        encoder_weights=None,
        in_channels=config["num_channels"],
        classes=config["num_classes"],
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True) # parameters loaded into the model in this line
    model.load_state_dict(state_dict)

    model.eval()
    return model.to(device)
