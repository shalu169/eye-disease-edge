# Fundus Screening App — Local Mode MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a fully offline Android Flutter app that captures a fundus image via a phone-mounted DIYretCAM, runs the existing trained MobileNetV3-small multi-label model on-device, shows per-disease confidence, renders a true Grad-CAM heatmap per disease, and logs each capture (image + session ID + predictions) for later export.

**Architecture:** A host-side Python script (`export_model.py`) converts the existing PyTorch checkpoint into a TFLite backbone (everything up to the last spatial feature map) plus a small JSON bundle of the classifier head's raw weights. The Flutter app runs the backbone through `tflite_flutter`, then reimplements the tiny head (GAP → 1×1 conv → hardswish → linear → sigmoid) by hand in Dart — including its analytic backward pass — because TFLite's runtime cannot do backprop and the head has a nonlinearity that breaks the plain-CAM shortcut. Same export script also emits numeric fixtures (real PyTorch forward/backward outputs on real sample images) that both a Python test and the Dart tests are graded against, so the Dart reimplementation is provably equivalent to the PyTorch original rather than just "looks right."

**Tech Stack:** Python 3.14 / PyTorch 2.14 / timm 1.0.30 / onnx / onnx2tf (host, in `eye-disease-edge-ai/.venv`), Flutter (stable channel) / Dart, `tflite_flutter`, `camera`, `image`, `path_provider`, `archive` (Flutter, in `eye-disease-edge-ai/mobile_app/`).

**Spec:** `eye-disease-edge-ai/docs/superpowers/specs/2026-09-27-fundus-screening-mobile-app-design.md` (sections 1-3 context/scope, 4 Grad-CAM rationale, 5-8 components/data flow/error handling/testing — this plan implements everything in that spec except section 4a and the Remote-mode bullets, which are a separate follow-up plan).

## Global Constraints

- Android only — no iOS target, no iOS-specific plugin config.
- Local mode makes zero network calls, ever — no package in `pubspec.yaml` may be a networking client.
- Preprocessing and model math in Dart must numerically match `baseline.py`'s `val_tfm` and the `run002_baseline_tuned_best_epoch15.pt` checkpoint — not "close enough," provably matched via the fixtures in Task 4.
- Grad-CAM must be true gradient-based Grad-CAM (per spec section 4's chosen approach), not Score-CAM, not plain CAM, not an occlusion heuristic.
- All new Flutter code lives under `eye-disease-edge-ai/mobile_app/`; all new Python export code lives under `eye-disease-edge-ai/` alongside `baseline.py`.
- Model bundle (`backbone.tflite`, `head_weights.json`, `preprocess_config.json`) ships as Flutter assets, loaded at app startup — never downloaded.

## Review Focus

- **Preprocessing mismatch (resize/normalize) between Dart and `baseline.py`'s `val_tfm`** silently changes every prediction without crashing anything — the single highest-risk item per spec section 8. Covered by Task 4's byte-level fixture comparison.
- **ONNX→TFLite conversion of the backbone silently reorders axes or approximates an op** (hardswish, the SE-block's global-pool+sigmoid+multiply) so the TFLite backbone's feature map diverges from PyTorch's `forward_features` output even though the conversion "succeeds" with no errors. Covered by Task 3's numeric fixture comparison.
- **Grad-CAM heatmap is degenerate** (all-zero after ReLU, or uniform/saturated) for a real image even though it matches a fixture computed from the same trained-but-under-confident model — a heatmap that's technically "correct" but useless to a viewer. Task 9 adds an explicit non-degeneracy assertion (heatmap has more than one distinct value after normalization) on a real sample image, not just fixture-matching.
- **Stale heatmap when switching diseases** — tapping disease B after already viewing disease A's heatmap must recompute, not redisplay A's cached overlay. Task 14 adds a widget test that taps two different disease rows in sequence and asserts the two rendered heatmaps differ.
- **Retake leaves an orphaned image file or a partial log record** — a capture that's retaken before commit must not appear in `listRecords()` and must not leave an image file with no matching record after `exportZip()`. Task 17 adds this as an explicit test, not just "handles retake" prose.

---

## Task 1: Flutter project scaffold

**Files:**
- Create: `eye-disease-edge-ai/mobile_app/` (via `flutter create`)
- Modify: `eye-disease-edge-ai/mobile_app/android/app/build.gradle` (package name, min SDK)
- Modify: `eye-disease-edge-ai/mobile_app/pubspec.yaml` (project metadata)

**Interfaces:**
- Produces: a buildable Flutter/Android project at `eye-disease-edge-ai/mobile_app/` that later tasks add code into.

- [ ] **Step 1: Create the Flutter project**

```bash
cd /Users/piuaggawal/Documents/my_workspace/eye-disease-edge-ai
flutter create --platforms=android --org com.eyediseaseedgeai --project-name fundus_screener mobile_app
```

- [ ] **Step 2: Set a real min SDK (camera + tflite_flutter need API 24+)**

Edit `mobile_app/android/app/build.gradle`, in the `defaultConfig` block, set:

```gradle
minSdkVersion 24
```

- [ ] **Step 3: Verify the empty scaffold builds**

Run: `cd mobile_app && flutter build apk --debug`
Expected: `Built build/app/outputs/flutter-apk/app-debug.apk` with no errors.

- [ ] **Step 4: Commit**

```bash
git -C /Users/piuaggawal/Documents/my_workspace add eye-disease-edge-ai/mobile_app
git -C /Users/piuaggawal/Documents/my_workspace commit -m "chore: scaffold fundus_screener Flutter app"
```

(If `/Users/piuaggawal/Documents/my_workspace` is not yet a git repo, run `git init` there first and confirm with the user before committing anything else in the workspace.)

---

## Task 2: Export script skeleton + checkpoint loader

**Files:**
- Create: `eye-disease-edge-ai/export_model.py`
- Create: `eye-disease-edge-ai/tests/test_export_model.py`
- Create: `eye-disease-edge-ai/tests/__init__.py` (empty)

**Interfaces:**
- Produces: `load_checkpoint(checkpoint_path: str) -> torch.nn.Module` — returns the 5-label `mobilenetv3_small_100` model in `eval()` mode with weights loaded.
- Produces: `LABEL_COLS = ["D", "G", "C", "A", "H"]` (module constant, re-exported for later tasks).

- [ ] **Step 1: Write the failing test**

```python
# eye-disease-edge-ai/tests/test_export_model.py
import torch
from export_model import load_checkpoint, LABEL_COLS

CHECKPOINT = "results/checkpoints/run002_baseline_tuned_best_epoch15.pt"


def test_load_checkpoint_returns_eval_model_with_right_output_size():
    model = load_checkpoint(CHECKPOINT)
    assert model.training is False
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, len(LABEL_COLS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# eye-disease-edge-ai/export_model.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eye-disease-edge-ai/export_model.py eye-disease-edge-ai/tests
git commit -m "feat: export_model.py skeleton, checkpoint loader"
```

---

## Task 3: Export the backbone to TFLite, with numeric parity test

**Files:**
- Modify: `eye-disease-edge-ai/export_model.py`
- Modify: `eye-disease-edge-ai/tests/test_export_model.py`
- Requires (host venv): `onnx`, `onnx2tf`, `tensorflow` — add to a new `eye-disease-edge-ai/requirements-export.txt`

**Interfaces:**
- Consumes: `load_checkpoint` from Task 2.
- Produces: `export_backbone_tflite(model: torch.nn.Module, out_dir: Path) -> Path` — writes `out_dir/backbone.tflite`, returns its path. TFLite input is `[1, 224, 224, 3]` float32 NHWC (ImageNet-normalized, NOT raw 0-255), output is `[1, 7, 7, 576]` float32 NHWC.

- [ ] **Step 1: Add export dependencies**

```
# eye-disease-edge-ai/requirements-export.txt
onnx==1.17.0
onnx2tf==1.26.3
tensorflow==2.18.0
```

```bash
cd eye-disease-edge-ai && .venv/bin/pip install -r requirements-export.txt
```

- [ ] **Step 2: Write the failing test**

```python
# append to eye-disease-edge-ai/tests/test_export_model.py
import numpy as np
import tensorflow as tf
from pathlib import Path
from export_model import export_backbone_tflite

FIXTURE_IMAGE_DIRS = ["data/preprocessed_images/0_left.jpg", "data/preprocessed_images/0_right.jpg"]


def _load_and_preprocess(path):
    from PIL import Image
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # [1,3,224,224] NCHW


def test_backbone_tflite_matches_pytorch_forward_features(tmp_path):
    model = load_checkpoint(CHECKPOINT)
    tflite_path = export_backbone_tflite(model, tmp_path)
    assert tflite_path.exists()

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    for img_path in FIXTURE_IMAGE_DIRS:
        x_nchw = _load_and_preprocess(img_path)
        with torch.no_grad():
            expected = model.forward_features(x_nchw).numpy()  # [1,576,7,7] NCHW

        x_nhwc = x_nchw.permute(0, 2, 3, 1).numpy().astype(np.float32)  # [1,224,224,3]
        interpreter.set_tensor(input_detail["index"], x_nhwc)
        interpreter.invoke()
        actual_nhwc = interpreter.get_tensor(output_detail["index"])  # [1,7,7,576]
        actual_nchw = np.transpose(actual_nhwc, (0, 3, 1, 2))

        assert actual_nchw.shape == expected.shape
        max_abs_diff = np.max(np.abs(actual_nchw - expected))
        assert max_abs_diff < 1e-2, f"backbone TFLite diverges from PyTorch by {max_abs_diff} on {img_path}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v -k backbone_tflite`
Expected: FAIL with `ImportError: cannot import name 'export_backbone_tflite'`

- [ ] **Step 4: Write the implementation**

```python
# append to eye-disease-edge-ai/export_model.py
import subprocess
from pathlib import Path


class _BackboneWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model.forward_features(x)


def export_backbone_tflite(model: torch.nn.Module, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "backbone.onnx"

    wrapper = _BackboneWrapper(model).eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["image"], output_names=["features"],
        opset_version=17, dynamic_axes=None,
    )

    tf_out_dir = out_dir / "backbone_tf"
    subprocess.run(
        ["onnx2tf", "-i", str(onnx_path), "-o", str(tf_out_dir), "-osd"],
        check=True,
    )

    produced = list(tf_out_dir.glob("*_float32.tflite"))
    if not produced:
        raise RuntimeError(f"onnx2tf did not produce a float32 tflite file in {tf_out_dir}")

    final_path = out_dir / "backbone.tflite"
    produced[0].replace(final_path)
    return final_path
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v -k backbone_tflite`
Expected: PASS. If it fails with a shape or large numeric divergence, first check `onnx2tf`'s console output for which op it approximated — `hardswish`/`hardsigmoid` and the SE-block's global-pool+sigmoid+multiply are the most likely culprits — and retry with `onnx2tf -i backbone.onnx -o backbone_tf -osd --disable_group_convolution_integration` before changing the model architecture itself.

- [ ] **Step 6: Commit**

```bash
git add eye-disease-edge-ai/export_model.py eye-disease-edge-ai/tests eye-disease-edge-ai/requirements-export.txt
git commit -m "feat: export backbone to TFLite with PyTorch parity test"
```

---

## Task 4: Export head weights, preprocessing config, and Dart-test fixtures

**Files:**
- Modify: `eye-disease-edge-ai/export_model.py`
- Modify: `eye-disease-edge-ai/tests/test_export_model.py`

**Interfaces:**
- Consumes: `load_checkpoint`, `LABEL_COLS`, `IMAGENET_MEAN`, `IMAGENET_STD`, `IMG_SIZE` from Tasks 2-3.
- Produces: `export_head_weights(model, out_path: Path) -> None` — writes JSON `{"conv_head_weight": [1024][576], "conv_head_bias": [1024], "classifier_weight": [5][1024], "classifier_bias": [5]}`.
- Produces: `export_preprocess_config(out_path: Path) -> None` — writes JSON `{"image_size": 224, "mean": [...], "std": [...]}`.
- Produces: `generate_fixtures(model, image_paths: list[str], fixtures_json_out: Path, images_out_dir: Path) -> None` — for each image: copies it into `images_out_dir`, and appends to the fixtures JSON an entry `{"image_file": "...", "pooled_vector": [576], "logits": [5], "probs": [5], "gradcam_alpha": {"D": [576], "G": [...], ...}}` where `gradcam_alpha[label]` is `d(logit_label)/d(pooled_vector)` computed via real `torch.autograd`.

- [ ] **Step 1: Write the failing tests**

```python
# append to eye-disease-edge-ai/tests/test_export_model.py
import json
from export_model import export_head_weights, export_preprocess_config, generate_fixtures


def test_head_weights_match_state_dict_exactly(tmp_path):
    model = load_checkpoint(CHECKPOINT)
    out_path = tmp_path / "head_weights.json"
    export_head_weights(model, out_path)

    data = json.loads(out_path.read_text())
    sd = model.state_dict()
    np.testing.assert_allclose(data["conv_head_weight"], sd["conv_head.weight"].squeeze(-1).squeeze(-1).numpy(), atol=1e-6)
    np.testing.assert_allclose(data["conv_head_bias"], sd["conv_head.bias"].numpy(), atol=1e-6)
    np.testing.assert_allclose(data["classifier_weight"], sd["classifier.weight"].numpy(), atol=1e-6)
    np.testing.assert_allclose(data["classifier_bias"], sd["classifier.bias"].numpy(), atol=1e-6)


def test_preprocess_config_matches_baseline_val_tfm(tmp_path):
    out_path = tmp_path / "preprocess_config.json"
    export_preprocess_config(out_path)
    data = json.loads(out_path.read_text())
    assert data["image_size"] == 224
    assert data["mean"] == [0.485, 0.456, 0.406]
    assert data["std"] == [0.229, 0.224, 0.225]


def test_fixtures_gradcam_alpha_matches_real_autograd(tmp_path):
    model = load_checkpoint(CHECKPOINT)
    images_out = tmp_path / "images"
    fixtures_json = tmp_path / "fixtures.json"
    generate_fixtures(model, FIXTURE_IMAGE_DIRS, fixtures_json, images_out)

    fixtures = json.loads(fixtures_json.read_text())
    assert len(fixtures) == len(FIXTURE_IMAGE_DIRS)

    entry = fixtures[0]
    x = _load_and_preprocess(FIXTURE_IMAGE_DIRS[0]).clone().requires_grad_(False)
    feat = model.forward_features(x)
    pooled = model.global_pool(feat).flatten(1)
    pooled_leaf = pooled.detach().clone().requires_grad_(True)
    head_out = model.conv_head(pooled_leaf.unsqueeze(-1).unsqueeze(-1))
    head_out = model.act2(head_out).flatten(1)
    logits = model.classifier(head_out)

    for i, label in enumerate(LABEL_COLS):
        model.zero_grad()
        if pooled_leaf.grad is not None:
            pooled_leaf.grad.zero_()
        logits[0, i].backward(retain_graph=True)
        expected_alpha = pooled_leaf.grad[0].numpy()
        np.testing.assert_allclose(entry["gradcam_alpha"][label], expected_alpha, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v -k "head_weights or preprocess_config or fixtures_gradcam"`
Expected: FAIL — `export_head_weights`, `export_preprocess_config`, `generate_fixtures` don't exist yet.

- [ ] **Step 3: Write the implementation**

```python
# append to eye-disease-edge-ai/export_model.py
import json
import shutil


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v`
Expected: PASS (all tests in the file so far).

- [ ] **Step 5: Commit**

```bash
git add eye-disease-edge-ai/export_model.py eye-disease-edge-ai/tests
git commit -m "feat: export head weights, preprocess config, and gradcam fixtures"
```

---

## Task 5: CLI entrypoint — run the full export into `mobile_app/`

**Files:**
- Modify: `eye-disease-edge-ai/export_model.py`
- Modify: `eye-disease-edge-ai/tests/test_export_model.py`

**Interfaces:**
- Consumes: everything from Tasks 2-4.
- Produces: `main(checkpoint_path: str, mobile_app_dir: str, fixture_images: list[str]) -> None` — writes `mobile_app_dir/assets/model/{backbone.tflite,head_weights.json,preprocess_config.json}` and `mobile_app_dir/test/fixtures/{model_fixtures.json,images/*.jpg}`. CLI: `python export_model.py --checkpoint <path> --mobile-app-dir <path>`.

- [ ] **Step 1: Write the failing test**

```python
# append to eye-disease-edge-ai/tests/test_export_model.py
from export_model import main as export_main


def test_main_writes_all_artifacts_into_mobile_app_dir(tmp_path):
    mobile_app_dir = tmp_path / "mobile_app"
    export_main(CHECKPOINT, str(mobile_app_dir), FIXTURE_IMAGE_DIRS)

    assert (mobile_app_dir / "assets" / "model" / "backbone.tflite").exists()
    assert (mobile_app_dir / "assets" / "model" / "head_weights.json").exists()
    assert (mobile_app_dir / "assets" / "model" / "preprocess_config.json").exists()
    assert (mobile_app_dir / "test" / "fixtures" / "model_fixtures.json").exists()
    for image_path in FIXTURE_IMAGE_DIRS:
        assert (mobile_app_dir / "test" / "fixtures" / "images" / Path(image_path).name).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v -k test_main_writes`
Expected: FAIL — `main` doesn't exist yet.

- [ ] **Step 3: Write the implementation**

```python
# append to eye-disease-edge-ai/export_model.py
import argparse


def main(checkpoint_path: str, mobile_app_dir: str, fixture_images) -> None:
    mobile_app_dir = Path(mobile_app_dir)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd eye-disease-edge-ai && .venv/bin/python -m pytest tests/test_export_model.py -v`
Expected: PASS (full suite).

- [ ] **Step 5: Run the real export into the actual app directory**

```bash
cd eye-disease-edge-ai
.venv/bin/python export_model.py --checkpoint results/checkpoints/run002_baseline_tuned_best_epoch15.pt --mobile-app-dir mobile_app
```

Expected: no errors; `mobile_app/assets/model/` and `mobile_app/test/fixtures/` now populated with real artifacts.

- [ ] **Step 6: Register assets in pubspec.yaml**

Edit `mobile_app/pubspec.yaml`, add under `flutter:`:

```yaml
  assets:
    - assets/model/backbone.tflite
    - assets/model/head_weights.json
    - assets/model/preprocess_config.json
```

- [ ] **Step 7: Commit**

```bash
git add eye-disease-edge-ai/export_model.py eye-disease-edge-ai/tests eye-disease-edge-ai/mobile_app/assets eye-disease-edge-ai/mobile_app/test/fixtures eye-disease-edge-ai/mobile_app/pubspec.yaml
git commit -m "feat: export_model.py CLI, run real export into mobile_app assets"
```

---

## Task 6: Dart data models (labels, predictions, session records)

**Files:**
- Create: `mobile_app/lib/models/labels.dart`
- Create: `mobile_app/lib/models/prediction_result.dart`
- Create: `mobile_app/lib/models/session_record.dart`
- Create: `mobile_app/test/models/session_record_test.dart`

**Interfaces:**
- Produces: `const List<String> kDiseaseLabels = ["D","G","C","A","H"];` and `const Map<String,String> kDiseaseNames` (human-readable) in `labels.dart`.
- Produces: `class PredictionResult { final Map<String, double> probabilities; PredictionResult(this.probabilities); }` in `prediction_result.dart`.
- Produces: `class SessionRecord { String sessionId; DateTime timestamp; String imageFileName; Map<String,double> predictions; String mode; String deviceTag; Map<String,dynamic> toJson(); factory SessionRecord.fromJson(Map<String,dynamic> json); }` in `session_record.dart`.

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/models/session_record_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/models/session_record.dart';

void main() {
  test('SessionRecord round-trips through JSON', () {
    final record = SessionRecord(
      sessionId: 'patient-07',
      timestamp: DateTime.utc(2026, 9, 27, 10, 30),
      imageFileName: 'capture_001.jpg',
      predictions: {'D': 0.12, 'G': 0.03, 'C': 0.44, 'A': 0.02, 'H': 0.01},
      mode: 'local',
      deviceTag: 's25ultra_diyretcam',
    );

    final json = record.toJson();
    final restored = SessionRecord.fromJson(json);

    expect(restored.sessionId, 'patient-07');
    expect(restored.timestamp, DateTime.utc(2026, 9, 27, 10, 30));
    expect(restored.imageFileName, 'capture_001.jpg');
    expect(restored.predictions['C'], 0.44);
    expect(restored.mode, 'local');
    expect(restored.deviceTag, 's25ultra_diyretcam');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/models/session_record_test.dart`
Expected: FAIL — package doesn't exist.

- [ ] **Step 3: Write the implementation**

```dart
// mobile_app/lib/models/labels.dart
const List<String> kDiseaseLabels = ['D', 'G', 'C', 'A', 'H'];

const Map<String, String> kDiseaseNames = {
  'D': 'Diabetic Retinopathy',
  'G': 'Glaucoma',
  'C': 'Cataract',
  'A': 'Age-related Macular Degeneration',
  'H': 'Hypertensive Retinopathy',
};
```

```dart
// mobile_app/lib/models/prediction_result.dart
class PredictionResult {
  final Map<String, double> probabilities;
  const PredictionResult(this.probabilities);
}
```

```dart
// mobile_app/lib/models/session_record.dart
class SessionRecord {
  final String sessionId;
  final DateTime timestamp;
  final String imageFileName;
  final Map<String, double> predictions;
  final String mode;
  final String deviceTag;

  SessionRecord({
    required this.sessionId,
    required this.timestamp,
    required this.imageFileName,
    required this.predictions,
    required this.mode,
    required this.deviceTag,
  });

  Map<String, dynamic> toJson() => {
        'sessionId': sessionId,
        'timestamp': timestamp.toIso8601String(),
        'imageFileName': imageFileName,
        'predictions': predictions,
        'mode': mode,
        'deviceTag': deviceTag,
      };

  factory SessionRecord.fromJson(Map<String, dynamic> json) => SessionRecord(
        sessionId: json['sessionId'] as String,
        timestamp: DateTime.parse(json['timestamp'] as String),
        imageFileName: json['imageFileName'] as String,
        predictions: Map<String, double>.from(json['predictions'] as Map),
        mode: json['mode'] as String,
        deviceTag: json['deviceTag'] as String,
      );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/models/session_record_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mobile_app/lib/models mobile_app/test/models
git commit -m "feat: SessionRecord, PredictionResult, disease label constants"
```

---

## Task 7: Preprocessing (resize + normalize) matching `val_tfm`

**Files:**
- Create: `mobile_app/lib/inference/preprocess_config.dart`
- Create: `mobile_app/lib/inference/preprocessing.dart`
- Create: `mobile_app/test/inference/preprocessing_test.dart`
- Add dependency: `image: ^4.2.0` to `pubspec.yaml`

**Interfaces:**
- Produces: `class PreprocessConfig { final int imageSize; final List<double> mean; final List<double> std; factory PreprocessConfig.fromJson(Map<String,dynamic> json); }`
- Produces: `Float32List preprocessImage(img.Image image, PreprocessConfig config)` — returns a flat NHWC `[1, imageSize, imageSize, 3]` float32 buffer, resize via bilinear interpolation, then `(pixel/255 - mean[c]) / std[c]` per channel, matching `baseline.py`'s `Resize((224,224))` + `ToTensor()` + `Normalize(mean, std)`.

- [ ] **Step 1: Write the failing test**

Uses the real exported preprocess config and a real fixture image from Task 5, compared against a Python-computed reference tensor dumped to JSON.

```python
# one-off, run manually to produce the fixture the Dart test reads — not part of the app or test suite:
# cd eye-disease-edge-ai && .venv/bin/python -c "
# import json
# from export_model import _preprocess_for_fixtures
# t = _preprocess_for_fixtures('data/preprocessed_images/0_left.jpg')[0]  # [3,224,224]
# nhwc = t.permute(1,2,0).tolist()
# json.dump(nhwc, open('mobile_app/test/fixtures/preprocessed_0_left.json','w'))
# "
```

- [ ] **Step 2: Run the one-off script from Step 1**

Run it exactly as shown. Expected: creates `mobile_app/test/fixtures/preprocessed_0_left.json` (a `[224,224,3]` nested list).

- [ ] **Step 3: Write the failing Dart test**

```dart
// mobile_app/test/inference/preprocessing_test.dart
import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/preprocess_config.dart';
import 'package:fundus_screener/inference/preprocessing.dart';

void main() {
  test('preprocessImage matches Python val_tfm within tolerance', () {
    final configJson = jsonDecode(
        File('assets/model/preprocess_config.json').readAsStringSync());
    final config = PreprocessConfig.fromJson(configJson);

    final image = img.decodeJpg(
        File('test/fixtures/images/0_left.jpg').readAsBytesSync())!;
    final actual = preprocessImage(image, config); // flat [1,224,224,3]

    final expectedNested = jsonDecode(
        File('test/fixtures/preprocessed_0_left.json').readAsStringSync())
        as List; // [224][224][3]

    var maxAbsDiff = 0.0;
    var idx = 0;
    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        for (var c = 0; c < 3; c++) {
          final expected = (expectedNested[y][x][c] as num).toDouble();
          final diff = (actual[idx] - expected).abs();
          if (diff > maxAbsDiff) maxAbsDiff = diff;
          idx++;
        }
      }
    }
    expect(maxAbsDiff, lessThan(0.05),
        reason: 'preprocessing diverges from Python val_tfm by $maxAbsDiff');
  });
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/inference/preprocessing_test.dart`
Expected: FAIL — `preprocess_config.dart`/`preprocessing.dart` don't exist.

- [ ] **Step 5: Write the implementation**

```dart
// mobile_app/lib/inference/preprocess_config.dart
class PreprocessConfig {
  final int imageSize;
  final List<double> mean;
  final List<double> std;

  const PreprocessConfig({
    required this.imageSize,
    required this.mean,
    required this.std,
  });

  factory PreprocessConfig.fromJson(Map<String, dynamic> json) {
    return PreprocessConfig(
      imageSize: json['image_size'] as int,
      mean: (json['mean'] as List).map((e) => (e as num).toDouble()).toList(),
      std: (json['std'] as List).map((e) => (e as num).toDouble()).toList(),
    );
  }
}
```

```dart
// mobile_app/lib/inference/preprocessing.dart
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'preprocess_config.dart';

Float32List preprocessImage(img.Image image, PreprocessConfig config) {
  final resized = img.copyResize(
    image,
    width: config.imageSize,
    height: config.imageSize,
    interpolation: img.Interpolation.linear,
  );

  final buffer = Float32List(config.imageSize * config.imageSize * 3);
  var idx = 0;
  for (var y = 0; y < config.imageSize; y++) {
    for (var x = 0; x < config.imageSize; x++) {
      final pixel = resized.getPixel(x, y);
      final r = pixel.r / 255.0;
      final g = pixel.g / 255.0;
      final b = pixel.b / 255.0;
      buffer[idx++] = (r - config.mean[0]) / config.std[0];
      buffer[idx++] = (g - config.mean[1]) / config.std[1];
      buffer[idx++] = (b - config.mean[2]) / config.std[2];
    }
  }
  return buffer;
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/inference/preprocessing_test.dart`
Expected: PASS. If it fails only slightly over tolerance, check `image` package's resize interpolation against PyTorch's default `Resize` (bilinear, antialias in newer torchvision) — widen tolerance to 0.08 only if the diff is uniformly small and not concentrated at edges (which would indicate a real off-by-one in resize).

- [ ] **Step 7: Commit**

```bash
git add mobile_app/lib/inference mobile_app/test/inference mobile_app/pubspec.yaml mobile_app/test/fixtures/preprocessed_0_left.json
git commit -m "feat: Dart preprocessing matching baseline.py val_tfm"
```

---

## Task 8: Head weights model + forward pass (confidences)

**Files:**
- Create: `mobile_app/lib/inference/head_weights.dart`
- Create: `mobile_app/lib/inference/feature_map.dart`
- Create: `mobile_app/lib/inference/head_math.dart`
- Create: `mobile_app/test/inference/head_math_forward_test.dart`

**Interfaces:**
- Produces: `class FeatureMap { final Float32List data; final int height, width, channels; double at(int y, int x, int c); FeatureMap(this.data, this.height, this.width, this.channels); }` — `data` is flat NHWC (matches Task 3's TFLite output layout).
- Produces: `class HeadWeights { final List<List<double>> convHeadWeight; final List<double> convHeadBias; final List<List<double>> classifierWeight; final List<double> classifierBias; final List<String> labels; factory HeadWeights.fromJson(Map<String,dynamic> json); }`
- Produces: `class HeadForwardResult { final Float64List pooled; final Float64List preActivation; final Float64List activated; final Float64List logits; final Map<String,double> probabilities; }`
- Produces: `HeadForwardResult headForward(FeatureMap features, HeadWeights weights)`
- Produces: `double hardswish(double x)`

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/inference/head_math_forward_test.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/inference/feature_map.dart';
import 'package:fundus_screener/inference/head_weights.dart';
import 'package:fundus_screener/inference/head_math.dart';

void main() {
  test('headForward reproduces PyTorch logits and probabilities', () {
    final weights = HeadWeights.fromJson(
        jsonDecode(File('assets/model/head_weights.json').readAsStringSync()));
    final fixtures = jsonDecode(
        File('test/fixtures/model_fixtures.json').readAsStringSync()) as List;

    for (final entry in fixtures) {
      final pooledExpected =
          (entry['pooled_vector'] as List).map((e) => (e as num).toDouble()).toList();

      // Build a FeatureMap whose spatial average equals pooledExpected exactly:
      // a 1x1 spatial feature map IS its own average, so this isolates head
      // math from the separate backbone/pooling numerics already covered by
      // Task 3's parity test.
      final data = Float32List.fromList(pooledExpected);
      final features = FeatureMap(data, 1, 1, pooledExpected.length);

      final result = headForward(features, weights);

      final expectedLogits =
          (entry['logits'] as List).map((e) => (e as num).toDouble()).toList();
      final expectedProbs =
          (entry['probs'] as List).map((e) => (e as num).toDouble()).toList();

      for (var i = 0; i < weights.labels.length; i++) {
        expect(result.logits[i], closeTo(expectedLogits[i], 1e-3));
        expect(result.probabilities[weights.labels[i]]!,
            closeTo(expectedProbs[i], 1e-3));
      }
    }
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/inference/head_math_forward_test.dart`
Expected: FAIL — none of the head_math files exist.

- [ ] **Step 3: Write the implementation**

```dart
// mobile_app/lib/inference/feature_map.dart
import 'dart:typed_data';

class FeatureMap {
  final Float32List data; // flat, NHWC: index = (y*width + x)*channels + c
  final int height;
  final int width;
  final int channels;

  FeatureMap(this.data, this.height, this.width, this.channels);

  double at(int y, int x, int c) => data[(y * width + x) * channels + c];

  /// Global average pool over spatial dims -> one value per channel.
  Float64List globalAveragePool() {
    final pooled = Float64List(channels);
    final count = height * width;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        for (var c = 0; c < channels; c++) {
          pooled[c] += at(y, x, c);
        }
      }
    }
    for (var c = 0; c < channels; c++) {
      pooled[c] /= count;
    }
    return pooled;
  }
}
```

```dart
// mobile_app/lib/inference/head_weights.dart
class HeadWeights {
  final List<List<double>> convHeadWeight; // [1024][576]
  final List<double> convHeadBias;         // [1024]
  final List<List<double>> classifierWeight; // [5][1024]
  final List<double> classifierBias;         // [5]
  final List<String> labels;

  HeadWeights({
    required this.convHeadWeight,
    required this.convHeadBias,
    required this.classifierWeight,
    required this.classifierBias,
    required this.labels,
  });

  factory HeadWeights.fromJson(Map<String, dynamic> json) {
    List<List<double>> to2D(dynamic raw) => (raw as List)
        .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
        .toList();
    List<double> to1D(dynamic raw) =>
        (raw as List).map((e) => (e as num).toDouble()).toList();

    return HeadWeights(
      convHeadWeight: to2D(json['conv_head_weight']),
      convHeadBias: to1D(json['conv_head_bias']),
      classifierWeight: to2D(json['classifier_weight']),
      classifierBias: to1D(json['classifier_bias']),
      labels: (json['labels'] as List).map((e) => e as String).toList(),
    );
  }
}
```

```dart
// mobile_app/lib/inference/head_math.dart
import 'dart:math';
import 'dart:typed_data';
import 'feature_map.dart';
import 'head_weights.dart';

double hardswish(double x) {
  if (x <= -3) return 0.0;
  if (x >= 3) return x;
  return x * (x + 3) / 6.0;
}

double hardswishDerivative(double x) {
  if (x <= -3) return 0.0;
  if (x >= 3) return 1.0;
  return (2 * x + 3) / 6.0;
}

double _sigmoid(double x) => 1.0 / (1.0 + exp(-x));

class HeadForwardResult {
  final Float64List pooled;        // [576]
  final Float64List preActivation; // [1024] (before hardswish)
  final Float64List activated;     // [1024] (after hardswish)
  final Float64List logits;        // [5]
  final Map<String, double> probabilities;

  HeadForwardResult({
    required this.pooled,
    required this.preActivation,
    required this.activated,
    required this.logits,
    required this.probabilities,
  });
}

HeadForwardResult headForward(FeatureMap features, HeadWeights weights) {
  final pooled = features.globalAveragePool(); // [576]

  final preAct = Float64List(weights.convHeadBias.length); // [1024]
  for (var m = 0; m < preAct.length; m++) {
    var sum = weights.convHeadBias[m];
    final row = weights.convHeadWeight[m];
    for (var k = 0; k < pooled.length; k++) {
      sum += row[k] * pooled[k];
    }
    preAct[m] = sum;
  }

  final activated = Float64List(preAct.length);
  for (var m = 0; m < preAct.length; m++) {
    activated[m] = hardswish(preAct[m]);
  }

  final logits = Float64List(weights.classifierBias.length); // [5]
  for (var c = 0; c < logits.length; c++) {
    var sum = weights.classifierBias[c];
    final row = weights.classifierWeight[c];
    for (var m = 0; m < activated.length; m++) {
      sum += row[m] * activated[m];
    }
    logits[c] = sum;
  }

  final probabilities = <String, double>{};
  for (var c = 0; c < weights.labels.length; c++) {
    probabilities[weights.labels[c]] = _sigmoid(logits[c]);
  }

  return HeadForwardResult(
    pooled: pooled,
    preActivation: preAct,
    activated: activated,
    logits: logits,
    probabilities: probabilities,
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/inference/head_math_forward_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mobile_app/lib/inference mobile_app/test/inference
git commit -m "feat: Dart head forward pass (GAP, conv_head, hardswish, classifier, sigmoid)"
```

---

## Task 9: Grad-CAM backward pass and heatmap

**Files:**
- Create: `mobile_app/lib/inference/gradcam.dart`
- Create: `mobile_app/test/inference/gradcam_test.dart`

**Interfaces:**
- Consumes: `HeadForwardResult`, `HeadWeights`, `hardswishDerivative` from Task 8; `FeatureMap` from Task 8.
- Produces: `Float64List gradCamAlpha(HeadForwardResult forward, HeadWeights weights, int classIndex)` — returns `d(logit_classIndex)/d(pooled)`, length 576.
- Produces: `List<List<double>> computeHeatmap(FeatureMap features, Float64List alpha)` — returns a `[height][width]` map: `ReLU(Σ_c alpha[c] * features.at(y,x,c))`, then min-max normalized to `[0,1]`.

- [ ] **Step 1: Write the failing tests**

```dart
// mobile_app/test/inference/gradcam_test.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/inference/feature_map.dart';
import 'package:fundus_screener/inference/head_weights.dart';
import 'package:fundus_screener/inference/head_math.dart';
import 'package:fundus_screener/inference/gradcam.dart';

void main() {
  late HeadWeights weights;
  late List fixtures;

  setUpAll(() {
    weights = HeadWeights.fromJson(
        jsonDecode(File('assets/model/head_weights.json').readAsStringSync()));
    fixtures = jsonDecode(
        File('test/fixtures/model_fixtures.json').readAsStringSync()) as List;
  });

  test('gradCamAlpha matches PyTorch autograd for every label', () {
    for (final entry in fixtures) {
      final pooled = (entry['pooled_vector'] as List)
          .map((e) => (e as num).toDouble())
          .toList();
      final features =
          FeatureMap(Float32List.fromList(pooled), 1, 1, pooled.length);
      final forward = headForward(features, weights);

      final expectedAlphaByLabel = entry['gradcam_alpha'] as Map;
      for (var i = 0; i < weights.labels.length; i++) {
        final label = weights.labels[i];
        final alpha = gradCamAlpha(forward, weights, i);
        final expected = (expectedAlphaByLabel[label] as List)
            .map((e) => (e as num).toDouble())
            .toList();
        for (var k = 0; k < alpha.length; k++) {
          expect(alpha[k], closeTo(expected[k], 1e-3));
        }
      }
    }
  });

  test('computeHeatmap on a real feature map is not degenerate', () {
    // Synthetic 7x7x4 feature map with real spatial variation, not a
    // fixture-only check — guards against a heatmap that's technically
    // fixture-correct but collapses to a flat map on real inputs.
    final data = Float32List(7 * 7 * 4);
    var i = 0;
    for (var y = 0; y < 7; y++) {
      for (var x = 0; x < 7; x++) {
        for (var c = 0; c < 4; c++) {
          data[i++] = (y == 3 && x == 3) ? 5.0 : 0.1 * (x + y);
        }
      }
    }
    final features = FeatureMap(data, 7, 7, 4);
    final alpha = Float64List.fromList([1.0, 0.5, -0.5, 0.2]);

    final heatmap = computeHeatmap(features, alpha);

    final flat = heatmap.expand((row) => row).toList();
    final distinctValues = flat.toSet().length;
    expect(distinctValues, greaterThan(1),
        reason: 'heatmap collapsed to a single value: $flat');
    expect(flat.reduce(max), closeTo(1.0, 1e-9));
    expect(flat.reduce(min), greaterThanOrEqualTo(0.0));
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/inference/gradcam_test.dart`
Expected: FAIL — `gradcam.dart` doesn't exist.

- [ ] **Step 3: Write the implementation**

```dart
// mobile_app/lib/inference/gradcam.dart
import 'dart:math';
import 'dart:typed_data';
import 'feature_map.dart';
import 'head_weights.dart';
import 'head_math.dart';

/// d(logit[classIndex]) / d(pooled), via chain rule through the two-layer
/// head: logit = classifier(hardswish(conv_head(pooled))).
Float64List gradCamAlpha(
    HeadForwardResult forward, HeadWeights weights, int classIndex) {
  final classifierRow = weights.classifierWeight[classIndex]; // [1024]
  final numHidden = classifierRow.length;
  final numPooled = forward.pooled.length;

  // d(logit)/d(activated) = classifierRow, then chain through hardswish'.
  final dLogitDPreAct = Float64List(numHidden);
  for (var m = 0; m < numHidden; m++) {
    dLogitDPreAct[m] =
        classifierRow[m] * hardswishDerivative(forward.preActivation[m]);
  }

  // d(logit)/d(pooled)[k] = sum_m dLogitDPreAct[m] * convHeadWeight[m][k]
  final alpha = Float64List(numPooled);
  for (var m = 0; m < numHidden; m++) {
    final row = weights.convHeadWeight[m];
    final grad = dLogitDPreAct[m];
    for (var k = 0; k < numPooled; k++) {
      alpha[k] += grad * row[k];
    }
  }
  return alpha;
}

List<List<double>> computeHeatmap(FeatureMap features, Float64List alpha) {
  final raw = List.generate(
      features.height, (_) => List<double>.filled(features.width, 0.0));

  for (var y = 0; y < features.height; y++) {
    for (var x = 0; x < features.width; x++) {
      var sum = 0.0;
      for (var c = 0; c < features.channels; c++) {
        sum += alpha[c] * features.at(y, x, c);
      }
      raw[y][x] = max(0.0, sum); // ReLU
    }
  }

  var minVal = double.infinity;
  var maxVal = -double.infinity;
  for (final row in raw) {
    for (final v in row) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }
  final range = (maxVal - minVal).abs() < 1e-9 ? 1.0 : (maxVal - minVal);

  return raw
      .map((row) => row.map((v) => (v - minVal) / range).toList())
      .toList();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/inference/gradcam_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mobile_app/lib/inference/gradcam.dart mobile_app/test/inference/gradcam_test.dart
git commit -m "feat: Grad-CAM analytic backward pass and heatmap computation"
```

---

## Task 10: Bilinear upsample + overlay rendering

**Files:**
- Create: `mobile_app/lib/inference/heatmap_overlay.dart`
- Create: `mobile_app/test/inference/heatmap_overlay_test.dart`

**Interfaces:**
- Consumes: output of `computeHeatmap` (Task 9), `image` package's `img.Image`.
- Produces: `img.Image renderOverlay(img.Image original, List<List<double>> heatmap, {double opacity = 0.5})` — bilinearly upsamples `heatmap` (small, e.g. 7×7) to `original`'s dimensions, maps each value through a red-hot colormap, alpha-blends onto a copy of `original`, returns the new image (does not mutate `original`).

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/inference/heatmap_overlay_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/heatmap_overlay.dart';

void main() {
  test('renderOverlay upsamples heatmap to image size and tints hot pixels', () {
    final original = img.Image(width: 28, height: 28);
    img.fill(original, color: img.ColorRgb8(50, 50, 50));

    // Heatmap hot only at bottom-right corner.
    final heatmap = List.generate(
        7, (y) => List<double>.generate(7, (x) => (y == 6 && x == 6) ? 1.0 : 0.0));

    final overlaid = renderOverlay(original, heatmap);

    expect(overlaid.width, 28);
    expect(overlaid.height, 28);

    final hotPixel = overlaid.getPixel(27, 27);
    final coldPixel = overlaid.getPixel(0, 0);

    // Hot corner should shift toward red relative to the untouched corner.
    expect(hotPixel.r, greaterThan(coldPixel.r));
    // Original image must not be mutated.
    expect(original.getPixel(27, 27).r, 50);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/inference/heatmap_overlay_test.dart`
Expected: FAIL — file doesn't exist.

- [ ] **Step 3: Write the implementation**

```dart
// mobile_app/lib/inference/heatmap_overlay.dart
import 'package:image/image.dart' as img;

double _bilinearSample(List<List<double>> map, double y, double x) {
  final h = map.length;
  final w = map[0].length;
  final y0 = y.floor().clamp(0, h - 1);
  final x0 = x.floor().clamp(0, w - 1);
  final y1 = (y0 + 1).clamp(0, h - 1);
  final x1 = (x0 + 1).clamp(0, w - 1);
  final fy = y - y0;
  final fx = x - x0;

  final top = map[y0][x0] * (1 - fx) + map[y0][x1] * fx;
  final bottom = map[y1][x0] * (1 - fx) + map[y1][x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

img.Image renderOverlay(img.Image original, List<List<double>> heatmap,
    {double opacity = 0.5}) {
  final result = img.Image.from(original);
  final srcH = heatmap.length;
  final srcW = heatmap[0].length;

  for (var y = 0; y < result.height; y++) {
    // Map output pixel to heatmap coordinate space.
    final srcY = (y / (result.height - 1)) * (srcH - 1);
    for (var x = 0; x < result.width; x++) {
      final srcX = (x / (result.width - 1)) * (srcW - 1);
      final value = _bilinearSample(heatmap, srcY, srcX).clamp(0.0, 1.0);

      final heatR = (255 * value).round();
      final heatG = (255 * (1 - value) * 0.3).round();
      const heatB = 0;

      final basePixel = original.getPixel(x, y);
      final blendedR = (basePixel.r * (1 - opacity) + heatR * opacity).round();
      final blendedG = (basePixel.g * (1 - opacity) + heatG * opacity).round();
      final blendedB = (basePixel.b * (1 - opacity) + heatB * opacity).round();

      result.setPixelRgb(x, y, blendedR, blendedG, blendedB);
    }
  }
  return result;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/inference/heatmap_overlay_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mobile_app/lib/inference/heatmap_overlay.dart mobile_app/test/inference/heatmap_overlay_test.dart
git commit -m "feat: bilinear heatmap upsample and overlay rendering"
```

---

## Task 11: TFLite backbone engine (device-realistic test)

**Files:**
- Create: `mobile_app/lib/inference/backbone_engine.dart`
- Create: `mobile_app/test/inference/backbone_engine_test.dart`
- Add dependency: `tflite_flutter: ^0.11.0` to `pubspec.yaml`

**Interfaces:**
- Consumes: `Float32List` from `preprocessImage` (Task 7), `FeatureMap` from Task 8.
- Produces: `class BackboneEngine { Future<void> load(String assetPath); FeatureMap run(Float32List preprocessedInput); void close(); }` — `load` reads the TFLite model from a Flutter asset; `run` invokes it and returns a `FeatureMap` for the fixed shape `[7, 7, 576]`.

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/inference/backbone_engine_test.dart
import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/backbone_engine.dart';
import 'package:fundus_screener/inference/preprocess_config.dart';
import 'package:fundus_screener/inference/preprocessing.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('BackboneEngine output matches PyTorch pooled_vector fixture', () async {
    final config = PreprocessConfig.fromJson(jsonDecode(
        File('assets/model/preprocess_config.json').readAsStringSync()));
    final fixtures = jsonDecode(
        File('test/fixtures/model_fixtures.json').readAsStringSync()) as List;

    final engine = BackboneEngine();
    await engine.load('assets/model/backbone.tflite');

    for (final entry in fixtures) {
      final image = img.decodeJpg(File(
              'test/fixtures/images/${entry['image_file']}')
          .readAsBytesSync())!;
      final input = preprocessImage(image, config);

      final features = engine.run(input);
      final pooled = features.globalAveragePool();

      final expectedPooled = (entry['pooled_vector'] as List)
          .map((e) => (e as num).toDouble())
          .toList();

      var maxAbsDiff = 0.0;
      for (var i = 0; i < pooled.length; i++) {
        final diff = (pooled[i] - expectedPooled[i]).abs();
        if (diff > maxAbsDiff) maxAbsDiff = diff;
      }
      expect(maxAbsDiff, lessThan(0.05),
          reason: 'backbone+pool diverges from fixture by $maxAbsDiff on ${entry['image_file']}');
    }

    engine.close();
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/inference/backbone_engine_test.dart`
Expected: FAIL — `backbone_engine.dart` doesn't exist / `tflite_flutter` not a dependency yet.

- [ ] **Step 3: Add the dependency**

```bash
cd mobile_app && flutter pub add tflite_flutter
```

- [ ] **Step 4: Write the implementation**

```dart
// mobile_app/lib/inference/backbone_engine.dart
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'feature_map.dart';

class BackboneEngine {
  Interpreter? _interpreter;

  Future<void> load(String assetPath) async {
    _interpreter = await Interpreter.fromAsset(assetPath);
  }

  FeatureMap run(Float32List preprocessedInput) {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('BackboneEngine.load() must complete before run()');
    }

    final input = preprocessedInput.reshape([1, 224, 224, 3]);
    final output = List.generate(
        1, (_) => List.generate(7, (_) => List.generate(7, (_) => List.filled(576, 0.0))));

    interpreter.run(input, output);

    final flat = Float32List(7 * 7 * 576);
    var idx = 0;
    for (var y = 0; y < 7; y++) {
      for (var x = 0; x < 7; x++) {
        for (var c = 0; c < 576; c++) {
          flat[idx++] = output[0][y][x][c];
        }
      }
    }
    return FeatureMap(flat, 7, 7, 576);
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/inference/backbone_engine_test.dart`
Expected: PASS. This test loads the real asset from disk (`assets/model/...`) rather than through Flutter's asset bundle, since `flutter test` doesn't build a real APK — that's why the test's file paths are relative to the project root, not `AssetBundle.load`. Note this asymmetry in the code review for Task 15, where the real app must use `Interpreter.fromAsset` against the bundled asset instead.

- [ ] **Step 6: Commit**

```bash
git add mobile_app/lib/inference/backbone_engine.dart mobile_app/test/inference/backbone_engine_test.dart mobile_app/pubspec.yaml mobile_app/pubspec.lock
git commit -m "feat: TFLite backbone engine with fixture-based parity test"
```

---

## Task 12: Session log storage (save, list, retake/discard, export)

**Files:**
- Create: `mobile_app/lib/storage/session_log.dart`
- Create: `mobile_app/test/storage/session_log_test.dart`
- Add dependencies: `path_provider: ^2.1.0`, `archive: ^3.6.0`, `csv: ^6.0.0` to `pubspec.yaml`

**Interfaces:**
- Produces: `class SessionLog { SessionLog(String rootDirectoryPath); Future<void> commitRecord(SessionRecord record, Uint8List imageBytes); Future<List<SessionRecord>> listRecords(); Future<File> exportZip(String outputPath); }`
- Behavior: `commitRecord` writes the image file and appends the record to a manifest — a capture that's never passed to `commitRecord` (i.e. a retake that's discarded before commit) leaves no trace. `exportZip` packages every committed image plus a `manifest.csv` (one row per record: sessionId, timestamp, imageFileName, predictions per label, mode, deviceTag).

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/storage/session_log_test.dart
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:archive/archive_io.dart';
import 'package:fundus_screener/models/session_record.dart';
import 'package:fundus_screener/storage/session_log.dart';

void main() {
  late Directory tempDir;

  setUp(() {
    tempDir = Directory.systemTemp.createTempSync('session_log_test');
  });

  tearDown(() {
    tempDir.deleteSync(recursive: true);
  });

  test('commitRecord persists record and image; listRecords returns it', () async {
    final log = SessionLog(tempDir.path);
    final record = SessionRecord(
      sessionId: 'patient-01',
      timestamp: DateTime.utc(2026, 9, 27),
      imageFileName: 'capture_a.jpg',
      predictions: {'D': 0.1, 'G': 0.2, 'C': 0.3, 'A': 0.4, 'H': 0.5},
      mode: 'local',
      deviceTag: 's25ultra_diyretcam',
    );

    await log.commitRecord(record, Uint8List.fromList([1, 2, 3]));
    final records = await log.listRecords();

    expect(records.length, 1);
    expect(records.first.sessionId, 'patient-01');
    expect(File('${tempDir.path}/images/capture_a.jpg').existsSync(), isTrue);
  });

  test('a retake that is never committed leaves no record and no orphan file', () async {
    final log = SessionLog(tempDir.path);
    // Simulate a retake: an image is captured to a scratch file, but
    // discarded (never passed to commitRecord) because the user retook it.
    final scratchFile = File('${tempDir.path}/scratch_retake.jpg');
    await scratchFile.writeAsBytes(Uint8List.fromList([9, 9, 9]));
    await scratchFile.delete(); // app deletes the scratch capture on retake

    final records = await log.listRecords();
    expect(records, isEmpty);
    expect(Directory('${tempDir.path}/images').existsSync() ?
        Directory('${tempDir.path}/images').listSync() : [], isEmpty);
  });

  test('exportZip packages every committed image plus a manifest.csv row per record', () async {
    final log = SessionLog(tempDir.path);
    await log.commitRecord(
      SessionRecord(
        sessionId: 'patient-01',
        timestamp: DateTime.utc(2026, 9, 27),
        imageFileName: 'capture_a.jpg',
        predictions: {'D': 0.1, 'G': 0.2, 'C': 0.3, 'A': 0.4, 'H': 0.5},
        mode: 'local',
        deviceTag: 's25ultra_diyretcam',
      ),
      Uint8List.fromList([1, 2, 3]),
    );

    final zipPath = '${tempDir.path}/export.zip';
    await log.exportZip(zipPath);

    final bytes = File(zipPath).readAsBytesSync();
    final archive = ZipDecoder().decodeBytes(bytes);
    final names = archive.files.map((f) => f.name).toList();

    expect(names, contains('manifest.csv'));
    expect(names, contains('images/capture_a.jpg'));

    final manifestFile = archive.findFile('manifest.csv')!;
    final manifestText = String.fromCharCodes(manifestFile.content as List<int>);
    expect(manifestText, contains('patient-01'));
    expect(manifestText, contains('capture_a.jpg'));
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/storage/session_log_test.dart`
Expected: FAIL — `session_log.dart` doesn't exist.

- [ ] **Step 3: Add dependencies**

```bash
cd mobile_app && flutter pub add path_provider archive csv
```

- [ ] **Step 4: Write the implementation**

```dart
// mobile_app/lib/storage/session_log.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive_io.dart';
import 'package:csv/csv.dart';
import '../models/labels.dart';
import '../models/session_record.dart';

class SessionLog {
  final String rootDirectoryPath;

  SessionLog(this.rootDirectoryPath);

  File get _manifestFile => File('$rootDirectoryPath/manifest.jsonl');
  Directory get _imagesDir => Directory('$rootDirectoryPath/images');

  Future<void> commitRecord(SessionRecord record, Uint8List imageBytes) async {
    if (!_imagesDir.existsSync()) {
      _imagesDir.createSync(recursive: true);
    }
    final imageFile = File('${_imagesDir.path}/${record.imageFileName}');
    await imageFile.writeAsBytes(imageBytes);

    final line = jsonEncode(record.toJson());
    await _manifestFile.writeAsString('$line\n', mode: FileMode.append);
  }

  Future<List<SessionRecord>> listRecords() async {
    if (!_manifestFile.existsSync()) return [];
    final lines = await _manifestFile.readAsLines();
    return lines
        .where((line) => line.trim().isNotEmpty)
        .map((line) => SessionRecord.fromJson(jsonDecode(line)))
        .toList();
  }

  Future<File> exportZip(String outputPath) async {
    final records = await listRecords();
    final encoder = ZipFileEncoder();
    encoder.create(outputPath);

    final rows = <List<dynamic>>[
      ['sessionId', 'timestamp', 'imageFileName', ...kDiseaseLabels, 'mode', 'deviceTag'],
    ];
    for (final record in records) {
      rows.add([
        record.sessionId,
        record.timestamp.toIso8601String(),
        record.imageFileName,
        ...kDiseaseLabels.map((label) => record.predictions[label]),
        record.mode,
        record.deviceTag,
      ]);
      final imagePath = '${_imagesDir.path}/${record.imageFileName}';
      if (File(imagePath).existsSync()) {
        encoder.addFile(File(imagePath), 'images/${record.imageFileName}');
      }
    }

    final manifestCsv = const ListToCsvConverter().convert(rows);
    encoder.addArchiveFile(ArchiveFile(
        'manifest.csv', manifestCsv.length, utf8.encode(manifestCsv)));
    encoder.close();

    return File(outputPath);
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/storage/session_log_test.dart`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mobile_app/lib/storage mobile_app/test/storage mobile_app/pubspec.yaml mobile_app/pubspec.lock
git commit -m "feat: SessionLog storage with commit/list/export, retake leaves no trace"
```

---

## Task 13: Session ID entry screen

**Files:**
- Create: `mobile_app/lib/screens/session_id_screen.dart`
- Create: `mobile_app/test/screens/session_id_screen_test.dart`

**Interfaces:**
- Produces: `class SessionIdScreen extends StatelessWidget { final void Function(String sessionId) onSubmit; }` — a text field + submit button; calls `onSubmit` with the trimmed, non-empty entered ID.

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/screens/session_id_screen_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/screens/session_id_screen.dart';

void main() {
  testWidgets('submitting a non-empty ID calls onSubmit with trimmed value',
      (tester) async {
    String? submitted;
    await tester.pumpWidget(MaterialApp(
      home: SessionIdScreen(onSubmit: (id) => submitted = id),
    ));

    await tester.enterText(find.byType(TextField), '  patient-07  ');
    await tester.tap(find.byType(ElevatedButton));
    await tester.pump();

    expect(submitted, 'patient-07');
  });

  testWidgets('submit button is disabled for an empty ID', (tester) async {
    var called = false;
    await tester.pumpWidget(MaterialApp(
      home: SessionIdScreen(onSubmit: (id) => called = true),
    ));

    await tester.tap(find.byType(ElevatedButton));
    await tester.pump();

    expect(called, isFalse);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/screens/session_id_screen_test.dart`
Expected: FAIL — screen doesn't exist.

- [ ] **Step 3: Write the implementation**

```dart
// mobile_app/lib/screens/session_id_screen.dart
import 'package:flutter/material.dart';

class SessionIdScreen extends StatefulWidget {
  final void Function(String sessionId) onSubmit;

  const SessionIdScreen({super.key, required this.onSubmit});

  @override
  State<SessionIdScreen> createState() => _SessionIdScreenState();
}

class _SessionIdScreenState extends State<SessionIdScreen> {
  final _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Patient / Session ID')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: const InputDecoration(labelText: 'Session ID'),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                final trimmed = _controller.text.trim();
                if (trimmed.isNotEmpty) {
                  widget.onSubmit(trimmed);
                }
              },
              child: const Text('Continue to capture'),
            ),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/screens/session_id_screen_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mobile_app/lib/screens/session_id_screen.dart mobile_app/test/screens/session_id_screen_test.dart
git commit -m "feat: session ID entry screen"
```

---

## Task 14: Result screen — confidences + per-disease Grad-CAM on tap

**Files:**
- Create: `mobile_app/lib/screens/result_screen.dart`
- Create: `mobile_app/test/screens/result_screen_test.dart`

**Interfaces:**
- Consumes: `PredictionResult`, `kDiseaseLabels`, `kDiseaseNames` (Task 6); `FeatureMap`, `HeadWeights`, `HeadForwardResult`, `headForward` (Task 8); `gradCamAlpha`, `computeHeatmap` (Task 9); `renderOverlay` (Task 10).
- Produces: `class ResultScreen extends StatefulWidget` with constructor params `{required img.Image originalImage, required FeatureMap features, required HeadWeights weights, required HeadForwardResult forward, required void Function(String disease) onDiseaseSelected, required void Function() onSave, required void Function() onRetake}`. Renders one row per label in `kDiseaseLabels` with its confidence; tapping a row computes and displays that disease's Grad-CAM overlay, replacing any previously shown overlay. A "Save & continue" button calls `onSave` (the only path that leads to `SessionLog.commitRecord`); a "Retake" button calls `onRetake` instead — per spec section 7, nothing is committed unless the user explicitly saves, so a blurry/bad capture is discarded with no trace, matching the "no orphan file, no partial record" guarantee `SessionLog` already provides.

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/screens/result_screen_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:fundus_screener/inference/feature_map.dart';
import 'package:fundus_screener/inference/head_weights.dart';
import 'package:fundus_screener/inference/head_math.dart';
import 'package:fundus_screener/models/labels.dart';
import 'package:fundus_screener/screens/result_screen.dart';

HeadWeights _fakeWeights() {
  return HeadWeights(
    convHeadWeight: List.generate(4, (_) => List.filled(3, 0.5)),
    convHeadBias: List.filled(4, 0.0),
    classifierWeight: [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.5, 0.5, 0.5, 0.5],
    ],
    classifierBias: List.filled(5, 0.0),
    labels: kDiseaseLabels,
  );
}

void main() {
  testWidgets('shows one row per disease and a distinct heatmap per tapped row',
      (tester) async {
    final weights = _fakeWeights();
    final data = Float32List.fromList([1.0, 2.0, -1.0]);
    // Distinct per-pixel values so tapping different disease rows can
    // plausibly produce different heatmaps.
    final features = FeatureMap(
        Float32List.fromList(List.generate(
            3 * 3 * 3, (i) => (i % 3 == 0) ? 1.0 : (i % 3 == 1 ? 2.0 : -1.0))),
        3, 3, 3);
    final forward = headForward(features, weights);
    final original = img.Image(width: 10, height: 10);

    String? selected;
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(
        originalImage: original,
        features: features,
        weights: weights,
        forward: forward,
        onDiseaseSelected: (d) => selected = d,
        onSave: () {},
        onRetake: () {},
      ),
    ));

    for (final label in kDiseaseLabels) {
      expect(find.text(kDiseaseNames[label]!), findsOneWidget);
    }

    await tester.tap(find.text(kDiseaseNames['D']!));
    await tester.pump();
    expect(selected, 'D');
    expect(find.byKey(const Key('gradcam_overlay_image')), findsOneWidget);
    final bytesD = (tester
            .widget<Image>(find.byKey(const Key('gradcam_overlay_image')))
            .image as MemoryImage)
        .bytes;

    await tester.tap(find.text(kDiseaseNames['G']!));
    await tester.pump();
    expect(selected, 'G');
    // Still exactly one overlay shown (the new one replaced the old one,
    // not stacked), AND its actual pixel content changed — guards against
    // a stale/cached heatmap from the previous tap being redisplayed.
    expect(find.byKey(const Key('gradcam_overlay_image')), findsOneWidget);
    final bytesG = (tester
            .widget<Image>(find.byKey(const Key('gradcam_overlay_image')))
            .image as MemoryImage)
        .bytes;
    expect(bytesD, isNot(equals(bytesG)));
  });

  testWidgets('Save button calls onSave and Retake calls onRetake, never both',
      (tester) async {
    final weights = _fakeWeights();
    final features = FeatureMap(
        Float32List.fromList(List.generate(3 * 3 * 3, (i) => (i % 3).toDouble())),
        3, 3, 3);
    final forward = headForward(features, weights);

    var saved = false;
    var retaken = false;
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(
        originalImage: img.Image(width: 10, height: 10),
        features: features,
        weights: weights,
        forward: forward,
        onDiseaseSelected: (_) {},
        onSave: () => saved = true,
        onRetake: () => retaken = true,
      ),
    ));

    await tester.tap(find.byKey(const Key('retake_button')));
    await tester.pump();
    expect(retaken, isTrue);
    expect(saved, isFalse);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/screens/result_screen_test.dart`
Expected: FAIL — `result_screen.dart` doesn't exist.

- [ ] **Step 3: Write the implementation**

```dart
// mobile_app/lib/screens/result_screen.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import '../inference/feature_map.dart';
import '../inference/head_weights.dart';
import '../inference/head_math.dart';
import '../inference/gradcam.dart';
import '../inference/heatmap_overlay.dart';
import '../models/labels.dart';

class ResultScreen extends StatefulWidget {
  final img.Image originalImage;
  final FeatureMap features;
  final HeadWeights weights;
  final HeadForwardResult forward;
  final void Function(String disease) onDiseaseSelected;
  final void Function() onSave;
  final void Function() onRetake;

  const ResultScreen({
    super.key,
    required this.originalImage,
    required this.features,
    required this.weights,
    required this.forward,
    required this.onDiseaseSelected,
    required this.onSave,
    required this.onRetake,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  Uint8List? _overlayPngBytes;

  void _selectDisease(String label) {
    final classIndex = widget.weights.labels.indexOf(label);
    final alpha = gradCamAlpha(widget.forward, widget.weights, classIndex);
    final heatmap = computeHeatmap(widget.features, alpha);
    final overlaid = renderOverlay(widget.originalImage, heatmap);

    setState(() {
      _overlayPngBytes = Uint8List.fromList(img.encodePng(overlaid));
    });
    widget.onDiseaseSelected(label);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Screening result')),
      body: Column(
        children: [
          if (_overlayPngBytes != null)
            Image.memory(_overlayPngBytes!, key: const Key('gradcam_overlay_image')),
          Expanded(
            child: ListView(
              children: kDiseaseLabels.map((label) {
                final prob = widget.forward.probabilities[label] ?? 0.0;
                return ListTile(
                  title: Text(kDiseaseNames[label]!),
                  trailing: Text('${(prob * 100).toStringAsFixed(1)}%'),
                  onTap: () => _selectDisease(label),
                );
              }).toList(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                OutlinedButton(
                  key: const Key('retake_button'),
                  onPressed: widget.onRetake,
                  child: const Text('Retake'),
                ),
                ElevatedButton(
                  key: const Key('save_button'),
                  onPressed: widget.onSave,
                  child: const Text('Save & continue'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/screens/result_screen_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mobile_app/lib/screens/result_screen.dart mobile_app/test/screens/result_screen_test.dart
git commit -m "feat: result screen with per-disease confidence and Grad-CAM on tap"
```

---

## Task 15: Capture screen (camera, torch, framing guide, shutter)

**Files:**
- Create: `mobile_app/lib/screens/capture_screen.dart`
- Create: `mobile_app/test/screens/capture_screen_test.dart`
- Add dependency: `camera: ^0.11.0` to `pubspec.yaml`

**Interfaces:**
- Produces: `abstract class CameraSource { Widget buildPreview(); Future<Uint8List> takePicture(); Future<void> setTorch(bool enabled); }` — an interface so the real `camera`-plugin-backed implementation can be swapped for a fake in tests (real camera hardware doesn't exist in `flutter test`).
- Produces: `class CaptureScreen extends StatefulWidget { final CameraSource cameraSource; final void Function(Uint8List imageBytes) onCaptured; }` — shows `cameraSource.buildPreview()`, a fixed framing-guide overlay (a centered circle outline sized to a typical fundus FOV crop), and a shutter button that calls `cameraSource.takePicture()` then `onCaptured`. Calls `cameraSource.setTorch(true)` on init.

- [ ] **Step 1: Write the failing test**

```dart
// mobile_app/test/screens/capture_screen_test.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/screens/capture_screen.dart';

class FakeCameraSource implements CameraSource {
  bool torchEnabled = false;
  bool pictureTaken = false;

  @override
  Widget buildPreview() => const ColoredBox(color: Colors.black, child: SizedBox.expand());

  @override
  Future<void> setTorch(bool enabled) async {
    torchEnabled = enabled;
  }

  @override
  Future<Uint8List> takePicture() async {
    pictureTaken = true;
    return Uint8List.fromList([42]);
  }
}

void main() {
  testWidgets('forces torch on at startup and shows a framing guide', (tester) async {
    final source = FakeCameraSource();
    await tester.pumpWidget(MaterialApp(
      home: CaptureScreen(cameraSource: source, onCaptured: (_) {}),
    ));
    await tester.pump();

    expect(source.torchEnabled, isTrue);
    expect(find.byKey(const Key('framing_guide')), findsOneWidget);
  });

  testWidgets('tapping the shutter takes a picture and calls onCaptured',
      (tester) async {
    final source = FakeCameraSource();
    Uint8List? captured;
    await tester.pumpWidget(MaterialApp(
      home: CaptureScreen(
        cameraSource: source,
        onCaptured: (bytes) => captured = bytes,
      ),
    ));
    await tester.pump();

    await tester.tap(find.byKey(const Key('shutter_button')));
    await tester.pumpAndSettle();

    expect(source.pictureTaken, isTrue);
    expect(captured, Uint8List.fromList([42]));
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/screens/capture_screen_test.dart`
Expected: FAIL — `capture_screen.dart` doesn't exist.

- [ ] **Step 3: Add the dependency**

```bash
cd mobile_app && flutter pub add camera
```

- [ ] **Step 4: Write the implementation**

```dart
// mobile_app/lib/screens/capture_screen.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';

abstract class CameraSource {
  Widget buildPreview();
  Future<Uint8List> takePicture();
  Future<void> setTorch(bool enabled);
}

class CaptureScreen extends StatefulWidget {
  final CameraSource cameraSource;
  final void Function(Uint8List imageBytes) onCaptured;

  const CaptureScreen({
    super.key,
    required this.cameraSource,
    required this.onCaptured,
  });

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  @override
  void initState() {
    super.initState();
    widget.cameraSource.setTorch(true);
  }

  Future<void> _onShutterPressed() async {
    final bytes = await widget.cameraSource.takePicture();
    widget.onCaptured(bytes);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          widget.cameraSource.buildPreview(),
          Center(
            child: Container(
              key: const Key('framing_guide'),
              width: 240,
              height: 240,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.greenAccent, width: 3),
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 32.0),
              child: FloatingActionButton(
                key: const Key('shutter_button'),
                onPressed: _onShutterPressed,
                child: const Icon(Icons.camera),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/screens/capture_screen_test.dart`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mobile_app/lib/screens/capture_screen.dart mobile_app/test/screens/capture_screen_test.dart mobile_app/pubspec.yaml mobile_app/pubspec.lock
git commit -m "feat: capture screen with injectable CameraSource, torch, framing guide"
```

---

## Task 16: Real camera source implementation (device-only, not unit-tested)

**Files:**
- Create: `mobile_app/lib/screens/plugin_camera_source.dart`
- Modify: `mobile_app/android/app/src/main/AndroidManifest.xml` (camera permission)

**Interfaces:**
- Produces: `class PluginCameraSource implements CameraSource` — wraps the `camera` package's `CameraController` against the first available back camera, implementing the same interface `CaptureScreen` already consumes from Task 15. No new test here: real camera hardware isn't available under `flutter test`; this class is exercised by the manual device test in Task 18.

- [ ] **Step 1: Add camera permission**

Edit `mobile_app/android/app/src/main/AndroidManifest.xml`, inside `<manifest>`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
```

- [ ] **Step 2: Write the implementation**

```dart
// mobile_app/lib/screens/plugin_camera_source.dart
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/widgets.dart';
import 'capture_screen.dart';

class PluginCameraSource implements CameraSource {
  final CameraController controller;

  PluginCameraSource(this.controller);

  static Future<PluginCameraSource> create() async {
    final cameras = await availableCameras();
    final backCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );
    final controller = CameraController(backCamera, ResolutionPreset.high, enableAudio: false);
    await controller.initialize();
    return PluginCameraSource(controller);
  }

  @override
  Widget buildPreview() => CameraPreview(controller);

  @override
  Future<void> setTorch(bool enabled) async {
    await controller.setFlashMode(enabled ? FlashMode.torch : FlashMode.off);
  }

  @override
  Future<Uint8List> takePicture() async {
    final file = await controller.takePicture();
    return file.readAsBytes();
  }

  Future<void> dispose() async {
    await controller.dispose();
  }
}
```

- [ ] **Step 3: Verify the app still builds with the new permission and class**

Run: `cd mobile_app && flutter build apk --debug`
Expected: builds with no errors (this class isn't wired into `main.dart` until Task 17, so this is purely a compile check).

- [ ] **Step 4: Commit**

```bash
git add mobile_app/lib/screens/plugin_camera_source.dart mobile_app/android/app/src/main/AndroidManifest.xml
git commit -m "feat: real camera source backed by the camera plugin"
```

---

## Task 17: Wire the full pipeline in `main.dart`, with startup error handling

**Files:**
- Modify: `mobile_app/lib/main.dart`
- Create: `mobile_app/lib/screens/model_error_screen.dart`
- Create: `mobile_app/test/model_load_test.dart`

**Interfaces:**
- Consumes: everything from Tasks 6-16.
- Produces: `Future<ModelBundle> loadModelBundle()` in `main.dart`, where `class ModelBundle { final BackboneEngine engine; final HeadWeights weights; final PreprocessConfig preprocessConfig; }` — throws a descriptive `Exception` if any asset is missing/unparseable, caught by `main()` to show `ModelErrorScreen` instead of a blank/crashed app.
- Produces: app flow `SessionIdScreen -> CaptureScreen -> (preprocess, backbone, head forward) -> ResultScreen -> (on disease tap: Grad-CAM) -> (a "Save & continue" action) -> SessionLog.commitRecord -> back to CaptureScreen for the next capture (same session ID)`.

- [ ] **Step 1: Write the failing test for the error path**

```dart
// mobile_app/test/model_load_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:fundus_screener/main.dart';

void main() {
  test('loadModelBundle throws a descriptive error for a missing asset path', () async {
    expect(
      () => loadModelBundle(modelDirOverride: 'assets/does_not_exist'),
      throwsA(isA<Exception>().having(
          (e) => e.toString(), 'message', contains('model'))),
    );
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile_app && flutter test test/model_load_test.dart`
Expected: FAIL — `loadModelBundle` doesn't exist yet.

- [ ] **Step 3: Write `model_error_screen.dart`**

```dart
// mobile_app/lib/screens/model_error_screen.dart
import 'package:flutter/material.dart';

class ModelErrorScreen extends StatelessWidget {
  final String message;

  const ModelErrorScreen({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text('Could not load the on-device model:\n$message',
                  textAlign: TextAlign.center),
            ],
          ),
        ),
      ),
    );
  }
}
```

- [ ] **Step 4: Write `main.dart`**

```dart
// mobile_app/lib/main.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'inference/backbone_engine.dart';
import 'inference/head_weights.dart';
import 'inference/head_math.dart';
import 'inference/preprocess_config.dart';
import 'inference/preprocessing.dart';
import 'models/session_record.dart';
import 'screens/capture_screen.dart';
import 'screens/model_error_screen.dart';
import 'screens/plugin_camera_source.dart';
import 'screens/result_screen.dart';
import 'screens/session_id_screen.dart';
import 'storage/session_log.dart';

class ModelBundle {
  final BackboneEngine engine;
  final HeadWeights weights;
  final PreprocessConfig preprocessConfig;

  ModelBundle(this.engine, this.weights, this.preprocessConfig);
}

Future<ModelBundle> loadModelBundle({String modelDirOverride = 'assets/model'}) async {
  try {
    final engine = BackboneEngine();
    await engine.load('$modelDirOverride/backbone.tflite');

    final headJsonString =
        await rootBundleLoadString('$modelDirOverride/head_weights.json');
    final weights = HeadWeights.fromJson(jsonDecode(headJsonString));

    final preprocessJsonString =
        await rootBundleLoadString('$modelDirOverride/preprocess_config.json');
    final preprocessConfig =
        PreprocessConfig.fromJson(jsonDecode(preprocessJsonString));

    return ModelBundle(engine, weights, preprocessConfig);
  } catch (e) {
    throw Exception('failed to load on-device model bundle: $e');
  }
}

Future<String> rootBundleLoadString(String assetPath) async {
  // Thin wrapper so tests can point at a non-existent path without touching
  // Flutter's real AssetBundle, which isn't available outside a widget test.
  final file = File(assetPath);
  if (!file.existsSync()) {
    throw Exception('asset not found: $assetPath');
  }
  return file.readAsString();
}

void main() {
  runApp(const FundusScreenerApp());
}

class FundusScreenerApp extends StatelessWidget {
  const FundusScreenerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fundus Screener',
      home: FutureBuilder<ModelBundle>(
        future: loadModelBundle(),
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Scaffold(body: Center(child: CircularProgressIndicator()));
          }
          if (snapshot.hasError) {
            return ModelErrorScreen(message: snapshot.error.toString());
          }
          return _SessionFlow(modelBundle: snapshot.data!);
        },
      ),
    );
  }
}

class _SessionFlow extends StatefulWidget {
  final ModelBundle modelBundle;
  const _SessionFlow({required this.modelBundle});

  @override
  State<_SessionFlow> createState() => _SessionFlowState();
}

class _SessionFlowState extends State<_SessionFlow> {
  String? _sessionId;
  SessionLog? _sessionLog;

  Future<void> _ensureSessionLog() async {
    if (_sessionLog != null) return;
    final dir = await getApplicationDocumentsDirectory();
    _sessionLog = SessionLog(dir.path);
  }

  Future<void> _onCaptured(Uint8List imageBytes, BuildContext context) async {
    final original = img.decodeImage(imageBytes)!;
    final input = preprocessImage(original, widget.modelBundle.preprocessConfig);
    final features = widget.modelBundle.engine.run(input);
    final forward = headForward(features, widget.modelBundle.weights);

    if (!context.mounted) return;
    late BuildContext resultContext;
    final shouldSave = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (ctx) {
        resultContext = ctx;
        return ResultScreen(
          originalImage: original,
          features: features,
          weights: widget.modelBundle.weights,
          forward: forward,
          onDiseaseSelected: (_) {},
          onSave: () => Navigator.of(resultContext).pop(true),
          onRetake: () => Navigator.of(resultContext).pop(false),
        );
      },
    ));

    // Only an explicit Save commits anything — per spec section 7, a
    // discarded/retaken capture (shouldSave == false or the user backing
    // out) must leave no record and no orphan file. Since the image bytes
    // only exist in memory until this point, "do nothing" already satisfies
    // that guarantee; SessionLog.commitRecord is the sole write path.
    if (shouldSave == true) {
      await _ensureSessionLog();
      final fileName = 'capture_${DateTime.now().microsecondsSinceEpoch}.jpg';
      await _sessionLog!.commitRecord(
        SessionRecord(
          sessionId: _sessionId!,
          timestamp: DateTime.now().toUtc(),
          imageFileName: fileName,
          predictions: forward.probabilities,
          mode: 'local',
          deviceTag: 's25ultra_diyretcam',
        ),
        Uint8List.fromList(img.encodeJpg(original)),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_sessionId == null) {
      return SessionIdScreen(onSubmit: (id) => setState(() => _sessionId = id));
    }

    return FutureBuilder<PluginCameraSource>(
      future: PluginCameraSource.create(),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return const Scaffold(body: Center(child: CircularProgressIndicator()));
        }
        return CaptureScreen(
          cameraSource: snapshot.data!,
          onCaptured: (bytes) => _onCaptured(bytes, context),
        );
      },
    );
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd mobile_app && flutter test test/model_load_test.dart`
Expected: PASS

- [ ] **Step 6: Run the full test suite**

Run: `cd mobile_app && flutter test`
Expected: PASS across every file created in Tasks 6-17.

- [ ] **Step 7: Build the release APK**

Run: `cd mobile_app && flutter build apk --release`
Expected: builds successfully — this is the artifact installed on the Samsung S25 Ultra in Task 18.

- [ ] **Step 8: Commit**

```bash
git add mobile_app/lib/main.dart mobile_app/lib/screens/model_error_screen.dart mobile_app/test/model_load_test.dart
git commit -m "feat: wire full capture -> inference -> result -> save pipeline in main.dart"
```

---

## Task 18: Manual device verification (Samsung S25 Ultra + DIYretCAM)

**Files:** none — this task produces no code, only a verification record.

**Interfaces:** none.

- [ ] **Step 1: Install the release APK**

```bash
adb install mobile_app/build/app/outputs/flutter-apk/app-release.apk
```

- [ ] **Step 2: Run the end-to-end flow with a real DIYretCAM capture**

1. Attach the DIYretCAM lens/tube/clip rig to the S25 Ultra.
2. Launch the app, enter a test session ID (e.g. `manual-test-01`).
3. Confirm the torch turns on automatically and the framing guide overlay is visible in the preview.
4. Capture a real fundus image of a test subject (or a printed fundus photo as a smoke test if no subject is available yet).
5. Confirm the result screen shows all 5 disease confidences within ~1-2 seconds.
6. Tap each of the 5 disease rows in turn; confirm the heatmap overlay visibly changes each time (not stuck on the first one).
7. Take a deliberately blurry second capture and tap **Retake**; confirm the app returns to the capture screen with no new entry added (check record count before/after via the method in step 8 below) — this is the manual confirmation of Task 12/14's automated no-orphan-record tests on a real device.
8. Repeat capture and this time tap **Save & continue**; confirm the capture was written to the session log (check via `adb shell run-as com.eyediseaseedgeai.fundus_screener ls files/images` or by triggering the export action once it exists — export UI itself is out of scope for this plan per the spec, so verifying via `adb shell` is acceptable here).
9. Force-close and relaunch the app; confirm the previously committed record is still present (`SessionLog.listRecords()` persists across restarts since it reads from disk, not memory).

- [ ] **Step 3: Record the result**

Add an entry to `eye-disease-edge-ai/results/EXPERIMENTS.md` following the existing run-log convention (see Run 001-003), noting: date, device (Samsung S25 Ultra), DIYretCAM rig used, whether the guide/torch/capture/inference/Grad-CAM/persistence steps above all passed, and any issues found (e.g. inference latency, heatmap quality on a real capture vs. the ODIR-5K fixture images it was numerically validated against).
