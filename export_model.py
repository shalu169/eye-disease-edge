"""
Converts the trained baseline checkpoint into the artifacts the Flutter
app (mobile_app/) consumes: a TFLite backbone, a head-weights bundle, a
preprocessing-constants file, and numeric fixtures used by both the
Python and Dart test suites. See
docs/superpowers/specs/2026-09-27-fundus-screening-mobile-app-design.md
sections 3-4 for why the model is split into backbone/head this way.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    a GitHub release asset. That asset 404s as of this writing, and the
    resulting error page is fed to np.load(), which raises ValueError
    ("Cannot load file containing pickled data") and aborts the whole
    conversion before it produces any output. onnx2tf looks for this exact
    filename in os.getcwd() before attempting the network call, so
    pre-seeding a syntactically valid file of the expected shape/dtype there
    (with the subprocess cwd set to `cwd`) avoids the crash.

    NOTE: this array is NOT purely informational logging input. onnx2tf feeds
    it into `test_data_nhwc` / `onnx_tensor_infos_for_validation`
    (onnx2tf/onnx2tf.py ~L1019-1072), which its default-on "automatic
    correction of accuracy degradation" logic (`disable_strict_mode` is left
    at its default False in our subprocess call, i.e. strict mode -- meaning
    this correction logic runs) can use to choose between candidate axis
    permutations when writing ops into the exported graph (see e.g.
    onnx2tf/ops/Conv.py ~L654-807). So this random array can, in principle,
    influence how onnx2tf shapes the converted graph, not just what it logs.
    We are not relying on this array being inert: correctness of the
    resulting backbone.tflite is established independently by
    test_backbone_tflite_matches_pytorch_forward_features, which compares
    real-image outputs against PyTorch's forward_features() end-to-end
    (measured max_abs_diff ~1e-4 against a 1e-2 tolerance) -- not by any
    property of this seeded calibration data.
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

    # Do all intermediate work (ONNX export, onnx2tf conversion) in a temporary
    # directory; only move the final backbone.tflite to out_dir to avoid
    # committing ~12MB of intermediate onnx2tf artifacts.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        onnx_path = tmpdir / "backbone.onnx"

        wrapper = _BackboneWrapper(model).eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        torch.onnx.export(
            wrapper, dummy, str(onnx_path),
            input_names=["image"], output_names=["features"],
            opset_version=17, dynamic_axes=None, dynamo=False,
        )

        tf_out_dir = tmpdir / "backbone_tf"
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

        # Only move the final tflite to out_dir; temporary onnx and SavedModel stay in tmpdir
        final_path = out_dir / "backbone.tflite"
        shutil.copy(produced[0], final_path)

    return final_path


def export_head_weights(model: torch.nn.Module, out_path) -> None:
    sd = model.state_dict()
    data = {
        "conv_head_weight": sd["conv_head.weight"].squeeze(-1).squeeze(-1).tolist(),  # [1024,576]
        "conv_head_bias": sd["conv_head.bias"].tolist(),                              # [1024]
        "classifier_weight": sd["classifier.weight"].tolist(),                        # [5,1024]
        "classifier_bias": sd["classifier.bias"].tolist(),                            # [5]
        "labels": LABEL_COLS,
    }
    Path(out_path).write_text(json.dumps(data))


def export_preprocess_config(out_path) -> None:
    data = {"image_size": IMG_SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    Path(out_path).write_text(json.dumps(data))


def _preprocess_for_fixtures(image_path):
    from PIL import Image
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)


def generate_fixtures(model: torch.nn.Module, image_paths, fixtures_json_out, images_out_dir) -> None:
    images_out_dir = Path(images_out_dir)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for image_path in image_paths:
        image_path = Path(image_path)
        dest_name = image_path.name
        shutil.copy(image_path, images_out_dir / dest_name)

        x = _preprocess_for_fixtures(image_path)
        feat = model.forward_features(x)
        pooled = model.global_pool(feat).flatten(1)
        pooled_leaf = pooled.detach().clone().requires_grad_(True)
        head_out = model.conv_head(pooled_leaf.unsqueeze(-1).unsqueeze(-1))
        head_out = model.act2(head_out).flatten(1)
        logits = model.classifier(head_out)
        probs = torch.sigmoid(logits)

        gradcam_alpha = {}
        for i, label in enumerate(LABEL_COLS):
            model.zero_grad()
            if pooled_leaf.grad is not None:
                pooled_leaf.grad.zero_()
            logits[0, i].backward(retain_graph=True)
            gradcam_alpha[label] = pooled_leaf.grad[0].detach().tolist()

        entries.append({
            "image_file": dest_name,
            "pooled_vector": pooled[0].detach().tolist(),
            "logits": logits[0].detach().tolist(),
            "probs": probs[0].detach().tolist(),
            "gradcam_alpha": gradcam_alpha,
        })

    Path(fixtures_json_out).write_text(json.dumps(entries))


def main(checkpoint_path: str, mobile_app_dir: str, fixture_images) -> None:
    mobile_app_dir = Path(mobile_app_dir).resolve()
    model_dir = mobile_app_dir / "assets" / "model"
    fixtures_dir = mobile_app_dir / "test" / "fixtures"
    images_dir = fixtures_dir / "images"

    model = load_checkpoint(checkpoint_path)
    export_backbone_tflite(model, model_dir)
    export_head_weights(model, model_dir / "head_weights.json")
    export_preprocess_config(model_dir / "preprocess_config.json")
    generate_fixtures(model, fixture_images, fixtures_dir / "model_fixtures.json", images_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/run002_baseline_tuned_best_epoch15.pt")
    parser.add_argument("--mobile-app-dir", default="mobile_app")
    parser.add_argument("--fixture-image", action="append", default=[
        "data/preprocessed_images/0_left.jpg",
        "data/preprocessed_images/0_right.jpg",
        "data/preprocessed_images/1005_right.jpg",
    ])
    args = parser.parse_args()
    main(args.checkpoint, args.mobile_app_dir, args.fixture_image)
