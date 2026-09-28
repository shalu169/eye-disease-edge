import torch
from export_model import load_checkpoint, LABEL_COLS
from baseline import LABEL_COLS as BASELINE_LABEL_COLS, IMG_SIZE as BASELINE_IMG_SIZE

CHECKPOINT = "results/checkpoints/run002_baseline_tuned_best_epoch15.pt"


def test_label_cols_matches_baseline():
    # export_model.py keeps its own copy of LABEL_COLS (it must be
    # importable/usable without pulling in baseline.py's heavier training
    # deps like pandas/sklearn at export time). If a future retrain changes
    # baseline.py's label set/order and export_model.py's copy isn't updated
    # to match, on-device predictions would be silently mislabeled (e.g.
    # displayed under the wrong disease name) even though every other test
    # stays green. This test is the guard against that drift.
    assert LABEL_COLS == BASELINE_LABEL_COLS


def test_load_checkpoint_returns_eval_model_with_right_output_size():
    model = load_checkpoint(CHECKPOINT)
    assert model.training is False
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, len(LABEL_COLS))


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

    # In addition to (not instead of) the hardcoded-literal checks above:
    # confirm export_model.py's IMG_SIZE actually still matches baseline.py's
    # real IMG_SIZE, so a future retrain that changes baseline.py's input
    # resolution can't silently desync from what gets exported to the app
    # while this test suite stays green. IMAGENET_MEAN/IMAGENET_STD have no
    # baseline.py equivalent to import against -- they're inline literals in
    # baseline.py's val_tfm/train_tfm (not module-level names), not
    # module-level constants there -- so those remain checked only against
    # the fixed literals above.
    assert BASELINE_IMG_SIZE == 224
    assert data["image_size"] == BASELINE_IMG_SIZE


def test_fixtures_gradcam_alpha_matches_real_autograd(tmp_path):
    model = load_checkpoint(CHECKPOINT)
    images_out = tmp_path / "images"
    fixtures_json = tmp_path / "fixtures.json"
    generate_fixtures(model, FIXTURE_IMAGE_DIRS, fixtures_json, images_out)

    fixtures = json.loads(fixtures_json.read_text())
    assert len(fixtures) == len(FIXTURE_IMAGE_DIRS)

    entry = fixtures[0]
    x = _load_and_preprocess(FIXTURE_IMAGE_DIRS[0]).clone().requires_grad_(False)

    # Cross-check: verify fixture logits match direct model forward pass (catches head-reconstruction drift)
    with torch.no_grad():
        full_model_logits = model(x).numpy()
    np.testing.assert_allclose(entry["logits"], full_model_logits[0], atol=1e-4)

    # Verify manual head reconstruction matches for gradcam computation
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
