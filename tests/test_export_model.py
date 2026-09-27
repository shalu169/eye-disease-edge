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
