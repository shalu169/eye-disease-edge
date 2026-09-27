"""
Converts the trained baseline checkpoint into the artifacts the Flutter
app (mobile_app/) consumes: a TFLite backbone, a head-weights bundle, a
preprocessing-constants file, and numeric fixtures used by both the
Python and Dart test suites. See
docs/superpowers/specs/2026-09-27-fundus-screening-mobile-app-design.md
sections 3-4 for why the model is split into backbone/head this way.
"""

import timm
import torch

LABEL_COLS = ["D", "G", "C", "A", "H"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=len(LABEL_COLS))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
