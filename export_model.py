"""
Converts the trained baseline checkpoint into the artifacts the Flutter
app (mobile_app/) consumes: a TFLite backbone, a head-weights bundle, a
preprocessing-constants file, and numeric fixtures used by both the
Python and Dart test suites. See
docs/superpowers/specs/2026-09-27-fundus-screening-mobile-app-design.md
sections 3-4 for why the model is split into backbone/head this way.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


class _BackboneWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model.forward_features(x)


def _ensure_onnx2tf_calibration_cache(cwd: Path) -> None:
    """onnx2tf unconditionally tries to download a small calibration/test
    image array (calibration_image_sample_data_20x128x128x3_float32.npy) from
    a GitHub release asset for its internal per-op accuracy-check logging.
    That asset 404s as of this writing, and the resulting error page is fed
    to np.load(), which raises ValueError ("Cannot load file containing
    pickled data") and aborts the whole conversion -- this is unrelated to
    our model. onnx2tf looks for this exact filename in os.getcwd() before
    attempting the network call, so pre-seeding a syntactically valid file of
    the expected shape/dtype there (with the subprocess cwd set to `cwd`)
    short-circuits the broken download without touching model conversion.
    """
    filename = "calibration_image_sample_data_20x128x128x3_float32.npy"
    cache_path = cwd / filename
    if not cache_path.exists():
        rng = np.random.default_rng(0)
        dummy = rng.random((20, 128, 128, 3), dtype=np.float32)
        np.save(cache_path, dummy, allow_pickle=False)


def export_backbone_tflite(model: torch.nn.Module, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "backbone.onnx"

    wrapper = _BackboneWrapper(model).eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["image"], output_names=["features"],
        opset_version=17, dynamic_axes=None, dynamo=False,
    )

    tf_out_dir = out_dir / "backbone_tf"
    tf_out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_onnx2tf_calibration_cache(tf_out_dir)

    # onnx2tf shells out to sibling console scripts (e.g. onnxsim) by bare
    # name; make sure the venv's bin dir (where sys.executable lives) is on
    # PATH for the subprocess regardless of whether the venv is "activated".
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    subprocess.run(
        [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(tf_out_dir), "-osd"],
        check=True,
        cwd=str(tf_out_dir),
        env=env,
    )

    produced = list(tf_out_dir.glob("*_float32.tflite"))
    if not produced:
        raise RuntimeError(f"onnx2tf did not produce a float32 tflite file in {tf_out_dir}")

    final_path = out_dir / "backbone.tflite"
    produced[0].replace(final_path)
    return final_path
