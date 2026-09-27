# Smartphone Fundus Screening App — Design Spec

Date: 2026-09-27
Status: approved by user, pending implementation plan

## 1. Context

This app is a third hardware tier for the `eye-disease-edge-ai` paper (see
`eye-disease-edge-ai/abstract.md` and `eye-disease-edge-ai/brainstorm.md`),
alongside the Intel N100 mini PC (OpenVINO) and Raspberry Pi 3B (TFLite)
tiers already in progress. It reuses the same trained model rather than
introducing a new one.

Study design this app must support:
- Capture fundus images with a Samsung S25 Ultra + DIYretCAM (a DIY lens
  rig: 20D lens + PVC pipe + clip, no electronics — see
  `Raju et al., Indian J Ophthalmol 2016`, and Vaughan 2024 survey
  section 2.2, which reports no quantitative accuracy for DIYretCAM — this
  study is establishing that baseline).
- Take images from ~10 subjects with both the phone+DIYretCAM rig and a
  state-of-the-art gold-standard fundus device.
- Compare phone on-device predictions against: (a) the same model run on
  N100/Pi3B/this Mac, (b) a large SOTA foundation model (RETFound and/or
  DINOv2, fine-tuned on the user's own GPU on IDRiD/APTOS/PAPILA — a
  separate, host-side pipeline, NOT part of this app), and (c) an
  ophthalmologist's manual grading (ground truth).
- Comparisons (a)-(c) mostly happen offline, outside the app, using
  images and predictions the app exports. Additionally, the app supports
  an opt-in Remote mode that calls a configurable host-side inference
  server live, for cases where a live side-by-side comparison during the
  study is more useful than a fully offline correlation pass. Local mode
  has no dependency on any of these other components existing yet;
  Remote mode depends on the reference server (section 4a) being
  reachable.

## 2. Scope

**In scope (v1):**
- Android-only Flutter app at `eye-disease-edge-ai/mobile_app/`.
- Guided camera capture for DIYretCAM use (torch on, framing guide,
  manual shutter).
- **Two inference modes, switchable in Settings, decided per experiment
  run:**
  - **Local (default)** — fully offline, on-device, reusing the existing
    trained MobileNetV3-small checkpoint
    (`results/checkpoints/run002_baseline_tuned_best_epoch15.pt`, ODIR-5K, mean AUC 0.8215,
    labels D=DR, G=glaucoma, C=cataract, A=AMD, H=hypertensive
    retinopathy). This is the true "works with no internet" field story.
  - **Remote** — sends the captured image to a configurable HTTP
    endpoint (URL entered in Settings) and displays whatever predictions
    come back, using the same UI/export/log code paths as Local. Depends
    on a reachable server; used for the controlled study, not the field
    story.
- Per-disease confidence (sigmoid probability) shown immediately after
  capture, in either mode.
- Per-disease Grad-CAM heatmap overlay, viewable by tapping any of the 5
  disease rows, in either mode (computed on-device for Local, computed
  server-side and returned for Remote).
- Patient/session ID entry per capture, and a local log (image +
  metadata + predictions + which mode/endpoint produced them) that can be
  exported off-device for the correlation/doctor-review study.
- A host-side export script (`eye-disease-edge-ai/export_model.py`) that
  converts the PyTorch checkpoint into the TFLite backbone + head-weights
  bundle the Local mode consumes.
- A host-side reference inference server
  (`eye-disease-edge-ai/inference_server.py`), model-swappable via
  config, for Remote mode. See section 4a.

**Explicitly out of scope for v1 (future work, not part of this plan):**
- iOS build.
- Automatic image-quality/blur/framing detection — user visually judges
  the preview and retakes if needed.
- Fine-tuning RETFound/DINOv2 themselves — that happens separately on the
  user's GPU; this plan only builds the server abstraction they'll plug
  into once fine-tuned (see section 4a).
- Running the reference server on the N100 (OpenVINO) or Pi3B (TFLite)
  tiers — those already have their own benchmarking scripts per
  `brainstorm.md`; the reference server here targets a PyTorch-capable
  host (e.g. the GPU box) and is not required to unify with that
  existing pipeline.
- The doctor-grading workflow and the cross-hardware/cross-model
  correlation analysis itself — this app only needs to produce
  well-structured exportable data for those to happen later.
- Score-CAM or other alternative explainability methods (Grad-CAM via
  backbone/head split was chosen for Local mode; see section 4).

## 3. Architecture

Three deliverables:

1. **`eye-disease-edge-ai/mobile_app/`** — the Flutter/Android app.
   Local mode: capture → on-device inference → Grad-CAM → local log, no
   network. Remote mode: capture → HTTP call to configured endpoint →
   same log, using the response in place of the local computation.
2. **`eye-disease-edge-ai/export_model.py`** — host-side, one-time (re-run
   whenever the model is retrained) script alongside `baseline.py`. Loads
   `results/checkpoints/run002_baseline_tuned_best_epoch15.pt` (timm `mobilenetv3_small_100`,
   5-way multi-label head) and produces:
   - a TFLite model of the backbone only, truncated before global average
     pooling, output shape `[1, 7, 7, 576]` for 224×224 input;
   - a binary/JSON bundle of the head weights: `conv_head` (1×1 conv,
     576→1024, followed by hardswish) and `classifier` (Linear,
     1024→5), plus the exact preprocessing constants (resize 224,
     ImageNet mean/std) so the app's preprocessing is generated from the
     same source of truth rather than hand-copied.
3. **`eye-disease-edge-ai/inference_server.py`** — host-side reference
   server for Remote mode, model-swappable via config. See section 4a.

## 4. Why the backbone/head split (Grad-CAM constraint)

`timm`'s `mobilenetv3_small_100` head is GAP → conv_head (1×1 conv +
hardswish) → classifier (Linear) — not plain GAP→FC. The hardswish
nonlinearity between pooling and the class logit means the standard
"weight spatial maps by final FC weights" CAM shortcut does not hold, and
true Grad-CAM needs a backward pass, which TFLite's runtime does not
support (inference-only).

Resolution: export the backbone only up to the last spatial feature map
(before GAP), and re-implement the tiny head (conv_head + classifier,
~1.5M parameters) by hand in Dart, including its analytic backward pass
(hardswish's derivative is closed-form). This gives:
- Confidences: one TFLite inference (backbone) + head forward (GAP +
  hardswish + linear + sigmoid) in Dart.
- Grad-CAM for a chosen disease: head backward pass (analytic, chain rule
  through 2 small layers) to get per-channel weights, then
  `ReLU(Σ weight_k × feature_map_k)`, normalized and bilinearly upsampled
  to the captured image size, rendered as an overlay.

This was chosen over Score-CAM (gradient-free but needs ~576 forward
passes per image, 10-30s on-phone, coarser) and over dropping Grad-CAM
entirely (would break the paper's existing explainability claim).

## 4a. Remote-mode server abstraction

`inference_server.py` is a small FastAPI app that loads one "backend" at
startup, chosen by a config file (e.g. `server_config.yaml`: `backend:
mobilenetv3_baseline`, `model_path: ...`). All backends implement the
same interface:

```
class ModelBackend(ABC):
    def predict(self, image) -> dict[str, float]       # {"D":0.12, "G":0.03, ...}
    def gradcam(self, image, disease: str) -> np.ndarray  # HxW heatmap, 0-1
```

Endpoints:
- `POST /predict` (image) → `{"predictions": {...}, "model_id": "..."}`.
- `POST /gradcam?disease=D` (same image) → heatmap for just that disease,
  matching the app's "tap a disease row" flow so Local and Remote behave
  identically from the UI's perspective.

v1 ships exactly one backend, `backends/mobilenetv3_baseline.py`,
wrapping the existing PyTorch checkpoint directly (real backward pass
available server-side, so no backbone/head-split hack is needed here —
that hack is a Local-mode-only, TFLite-only workaround). This makes
Remote mode with this backend "the same model, different hardware" for
comparison against the Local (on-phone) prediction. Adding
`backends/retfound.py` or `backends/dinov2.py` later, once fine-tuned
elsewhere, is a config change plus one new backend file — no server or
app changes.

Not built now: an OpenVINO or TFLite backend to unify with the N100/Pi3B
tiers — those keep their own scripts per `brainstorm.md` (see section 2).

## 5. App components

- **Capture screen** — camera preview, torch forced on, fixed-size
  framing guide overlay, manual shutter (no autofocus assist beyond the
  phone's own, since DIYretCAM has no optics/electronics of its own
  beyond the lens/tube/clip).
- **Preprocessing** — resize 224×224, ImageNet mean/std normalize,
  matching `baseline.py`'s `val_tfm` exactly (no augmentation). Sourced
  from the constants exported by `export_model.py`, not hand-duplicated.
- **Inference engine** — `tflite_flutter` runs the backbone; a Dart
  module runs GAP → head forward for confidences, and head backward (on
  demand, per tapped disease) for Grad-CAM weights.
- **Grad-CAM renderer** — per selected disease: weighted feature-map
  sum → ReLU → normalize → bilinear upsample → overlay on the captured
  image. In Remote mode, the heatmap array comes from `/gradcam` instead
  of being computed locally; rendering/overlay code is shared.
- **Settings screen** — Local/Remote toggle, endpoint URL field
  (Remote only). Mode is a per-session setting, recorded on every record
  so exported data shows which mode/endpoint produced it.
- **Remote client** — on capture, if Remote is selected: `POST /predict`
  with the image, show confidences; on disease-row tap, `POST
  /gradcam?disease=X` with the same image, show the returned heatmap.
  Same downstream code as Local from this point on.
- **Session/export** — prompts for a patient/session ID before each
  capture. Each record: original image, timestamp, session ID, per
  disease probability (D/G/C/A/H), device tag `s25ultra_diyretcam`, mode
  (`local` or `remote:<url>`). Stored locally; a share/export action
  packages records (images + a CSV/JSON manifest) for pulling off-device
  into the paper's `results/EXPERIMENTS.md` logging convention.

## 6. Data flow

Capture → preprocess →
- **Local:** backbone TFLite inference → head forward (confidences shown
  immediately) → user taps a disease row → head backward + Grad-CAM
  overlay for that class.
- **Remote:** `POST /predict` (confidences shown when response arrives)
  → user taps a disease row → `POST /gradcam?disease=X` → overlay for
  that class.
→ record saved to session log (including mode) → repeat or export.

## 7. Error handling

- Blurry/out-of-frame capture: no automatic detection in v1; user judges
  the live preview and can retake before the record is committed to the
  log (retake does not silently overwrite an already-committed record).
- Model bundle missing/corrupt at app startup (Local mode): fail loud
  with a clear error screen — never silently fall back to a wrong or
  stale model.
- Remote mode network/server failure (unreachable endpoint, timeout, bad
  response): show a clear error on that capture, do not write a
  half-formed record to the log, let the user retry or switch to Local
  without losing the captured image.
- Local mode has no network dependency to fail on; that guarantee only
  holds when Local mode is selected.

## 8. Testing

- Unit test: Dart preprocessing output matches a fixture generated by
  Python's `val_tfm` on the same source image, within float tolerance —
  guards the most likely silent-correctness bug (preprocessing mismatch).
- Unit test: Dart head forward pass reproduces the PyTorch model's final
  sigmoid outputs on a handful of fixture images, within float tolerance
  — validates the backbone/head split preserved model behavior exactly.
- Unit test (server): `mobilenetv3_baseline` backend's `predict()` output
  matches `baseline.py`'s model on the same fixture images — confirms the
  reference server isn't a second, drifted copy of the model logic.
- Manual device test: end-to-end run on the Samsung S25 Ultra with a real
  DIYretCAM capture in both Local and Remote mode before v1 is considered
  done.

## 9. Open follow-up work (not part of this plan)

- Host-side fine-tuning of RETFound and/or DINOv2 on IDRiD/APTOS/PAPILA,
  then adding them as new `inference_server.py` backends for the SOTA
  comparison arm.
- The correlation analysis comparing phone/N100/Pi3B/SOTA-model/doctor
  outputs across the 10-subject study.
- Any image-quality auto-detection, iOS support, alternative
  explainability methods, or an OpenVINO/TFLite backend to unify the
  server with the N100/Pi3B tiers.
