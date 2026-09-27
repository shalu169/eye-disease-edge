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
