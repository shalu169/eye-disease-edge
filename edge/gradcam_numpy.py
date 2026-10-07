"""
Run 010: on-device classifier head + analytic Grad-CAM in plain NumPy.

The deployed backbone (ONNX / TFLite) outputs only the last feature map
A = forward_features(x) (576 channels, 16x16 at 512 px). Everything after it
is small enough to do in NumPy, with no autodiff on the device:

  pooled = mean_ij A_ij                       [576]   (GAP)
  z      = W1 pooled + b1                     [1024]  (conv_head, 1x1 conv = dense)
  h      = hardswish(z)
  logits = W2 h + b2                          [5]
  probs  = sigmoid(logits)

Because GAP is linear and the head after it is a 2-layer MLP, the Grad-CAM
gradient is the same at every spatial position:
  d logit_c / d A_k,ij = (1/HW) * d logit_c / d pooled_k
  d logit_c / d pooled = ( W2[c] * hardswish'(z) ) @ W1        [576]
  Grad-CAM weight w_c,k = mean_ij d logit_c / d A_k,ij = (1/HW) d logit_c / d pooled_k
  L_c = ReLU( sum_k w_c,k A_k )               [h, w]
then bilinear upsampling to the input size (same kernel as torch
F.interpolate(mode="bilinear", align_corners=False), done with two small
interpolation matrices) and division by the per-map max.

Idea and parity-test approach from the Flutter app (fundus-app-local-mode
worktree, lib/inference/gradcam.dart); this is the NumPy port. Python 3.9 /
numpy 1.26 compatible (Raspberry Pi 3B env).
"""
import numpy as np


def hardswish(z):
    return z * np.clip(z + 3.0, 0.0, 6.0) / 6.0


def hardswish_grad(z):
    # torch hardswish_backward: 0 for z < -3, z/3 + 0.5 for -3 <= z <= 3, 1 for z > 3
    return np.where(z < -3.0, 0.0, np.where(z <= 3.0, z / 3.0 + 0.5, 1.0))


def load_head(path):
    d = np.load(path)
    return {k: d[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2")}


def head_forward(feat_hwc, head):
    """feat_hwc: [h, w, C] float. Returns (probs [5], logits [5], z [1024])."""
    pooled = feat_hwc.reshape(-1, feat_hwc.shape[-1]).astype(np.float64).mean(0)
    z = head["W1"] @ pooled + head["b1"]
    logits = head["W2"] @ hardswish(z) + head["b2"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs, logits, z


def gradcam_weights(z, head, labels, hw):
    """Grad-CAM channel weights [L, C] = (1/HW) d logit / d pooled."""
    g = head["W2"][list(labels)] * hardswish_grad(z)[None, :]  # [L, 1024]
    return (g @ head["W1"]) / float(hw)  # [L, C]


def bilinear_matrix(n_in, n_out):
    """[n_out, n_in] matrix reproducing torch bilinear, align_corners=False."""
    scale = n_in / float(n_out)
    src = (np.arange(n_out) + 0.5) * scale - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.minimum(np.floor(src).astype(int), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    l1 = src - i0
    M = np.zeros((n_out, n_in))
    M[np.arange(n_out), i0] += 1.0 - l1
    M[np.arange(n_out), i1] += l1
    return M


class GradCAM:
    """Head + Grad-CAM on a feature map from any runtime."""

    def __init__(self, head_path, out_size=512, dtype=np.float32):
        self.head = load_head(head_path)
        self.out_size = out_size
        self.dtype = dtype
        self._mats = {}

    def _up(self, h, w):
        key = (h, w)
        if key not in self._mats:
            self._mats[key] = (bilinear_matrix(h, self.out_size).astype(self.dtype),
                               bilinear_matrix(w, self.out_size).astype(self.dtype))
        return self._mats[key]

    def predict(self, feat_hwc):
        return head_forward(feat_hwc, self.head)[0]

    def __call__(self, feat_hwc, labels=(0, 1, 2, 3, 4), upsample=True, normalise=True):
        """Returns probs [5] and maps [L, S, S] (or raw low-res [L, h, w] if upsample=False)."""
        h, w, c = feat_hwc.shape
        probs, _, z = head_forward(feat_hwc, self.head)
        wts = gradcam_weights(z, self.head, labels, h * w).astype(self.dtype)  # [L, C]
        lr = feat_hwc.reshape(h * w, c).astype(self.dtype) @ wts.T  # [hw, L]
        lr = np.maximum(lr, 0.0).T.reshape(len(labels), h, w)
        if not upsample:
            return probs, lr
        My, Mx = self._up(h, w)
        maps = np.einsum("yi,lij,xj->lyx", My, lr, Mx, optimize=True)
        if normalise:
            mx = maps.reshape(len(labels), -1).max(1)
            maps *= (1.0 / np.where(mx > 0, mx, 1.0)).astype(maps.dtype)[:, None, None]
        return probs, maps
