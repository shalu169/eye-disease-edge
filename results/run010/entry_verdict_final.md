- **Device conditions**:
  - **N100**: idle before every config (load avg ≤ 0.26, 3.2 GHz). The suite was
    run twice. Pass 1 used numpy's default OpenBLAS threading
    (`results/n100_run010/openblas_default/`). Pass 2, the primary one, used
    `OPENBLAS_NUM_THREADS=1` (`results/n100_run010/`). A first attempt failed
    before timing because `/usr/bin/time` is not installed on the N100 (rc 127,
    `out_attempt1_no_usr_bin_time/`); the driver now skips it there.
  - **Pi**: `get_throttled` was 0x50005 or 0x50000 and the ARM clock was
    600 MHz at every before/after sample (14:03-14:09 CEST), 48-55 °C. As in
    runs 006/009, all Pi latencies are under the ~600 MHz under-voltage cap,
    roughly 2x pessimistic.
  - **Pi first attempt** (`results/pi_run010/out_attempt1_openblas_default/`),
    default OpenBLAS threads: `plain` and `head` completed. `gradcam1` died with
    SIGSEGV after 74 s. `gradcam5` was still running after 16 min at ~390 % CPU
    (about 70 s expected) and was killed. 10-iteration probes with 1 and 4
    OpenBLAS threads both completed and gave the same maps, so the failure
    appears only in long runs with 4 OpenBLAS threads competing with the 4
    TFLite/XNNPACK threads on 4 cores. The cause was not pinned down. The final
    Pi suite used `OPENBLAS_NUM_THREADS=1`, and all 4 configs completed.

- **Artifacts**:
  - Scripts: `results/run010_common.py`, `run010_check_reference.py`,
    `run010_export_edge.py`, `run010_export_tflite.py`, `run010_parity_edge.py`,
    `run010_sanity.py`, `run010_train_randlabels.py` (not run),
    `run010_faithfulness.py`, `run010_idrid_prep.py`, `run010_localisation.py`,
    `run010_make_device_fixture.py`, `run010_figures.py`, `run010_tables.py`
    (generates every table above from the JSONs), `run010_run_final.sh`. Edge
    code: `edge/gradcam_numpy.py`, `edge/gradcam_bench.py`, `edge/run010_device.sh`.
  - Results (`results/run010/`, `*_final*`; development pass `*_dev*`):
    `reference_vs_pytorch_grad_cam_final.json`, `parity_{onnx,tflite}_final.json`,
    `sanity_final.json`, `faithfulness_final.json` (+ `_per_image.json`,
    `_curves.npz`), `localisation_final.json` (+ `_per_image.csv`),
    `idrid_manifest.csv`, `figure_selection_final.json`,
    `figure_idrid_selection_final.json`, `probs_final_{val,test}.npy`,
    `tables_final.md`, `log_*_final.txt`, `log_final_driver.txt`,
    `env_venv_run010_freeze.txt`.
  - Device results: `results/n100_run010/` and `results/pi_run010/`
    (`results_<runtime>_<mode>.json`, `log_*.txt`, `suite.log`, failed
    attempts in sub-directories). On the devices: `~/eye-edge/run010/`.
  - Edge artifacts: `edge/artifacts/run010_final_*`,
    `edge/artifacts/run010_bundle_final/`.
  - Figures (PDF + PNG preview): `paper/figs/run010_gradcam_examples`,
    `run010_idrid_overlays`, `run010_deletion_insertion`,
    `run010_sanity_cascade`. Colour map viridis (perceptually uniform, CVD-safe),
    each with a colourbar. Colour = positive Grad-CAM evidence for the named
    label scaled to that image's maximum: 0 (dark purple) = none, 1 (yellow) =
    strongest. It is not a probability. Example selection (seed 2026, uniform
    within each category; candidates: TP D 253, G 55, C 50, A 34, H 34; DR FP 360;
    other-disease FP 1451; mild-NPDR misses 22): `4354_left` (DR TP, moderate
    NPDR), `1294_right` (G), `726_right` (C), `608_left` (A), `4214_right` (H),
    `3431_right` (DR FP), `3110_right` (glaucoma FP), `4542_right` (mild NPDR
    missed). IDRiD overlays (seed 2027): IDRiD_01, 12, 32, 50.
  - IDRiD data: `data/idrid/` (zip, sha256, Kaggle metadata, `prep512/`).

- **Verdict (final model)**:
  - **The edge Grad-CAM is the same Grad-CAM, and no autodiff is needed on the
    device.**
    - The reference matches pytorch-grad-cam to 1.5e-6.
    - Mac, 300 test images x 5 labels: the ONNX backbone + NumPy head gives
      heatmaps within 2.3e-4 of PyTorch autograd (max abs diff, maps in [0,1])
      and probabilities within 1.1e-5. TFLite: 7.2e-4 (99th percentile 1.6e-4)
      and 6.8e-5. The hottest pixel is identical in 1499 of 1500 maps for both.
    - The NumPy maths alone (on torch features) adds ≤ 1.4e-5. The remaining
      error is the backbone runtime's.
    - On the devices, against the stored Mac reference: N100 maps ≤ 2.5e-4,
      probabilities ≤ 9.2e-6 (ORT) / 1.9e-6 (OpenVINO); Pi maps ≤ 3.9e-4,
      probabilities ≤ 1.2e-4. These diffs include the float16 storage of the
      reference maps.
  - **Passes the model-randomisation sanity check (Adebayo).**
    - Re-initialising only the classifier drops the median Spearman ρ between
      the randomised and original map to 0.075. That is already below the 0.144
      between the same label's maps of two *different* images.
    - Every deeper cascade stage stays between −0.045 and 0.022. Randomising only
      the Grad-CAM layer (blocks.5) gives 0.067.
    - SSIM stays at 0.22-0.48, below the different-image level of 0.53. On these
      smooth, mostly empty maps SSIM is dominated by background, so ρ is the
      informative metric.
    - The maps are class-specific: the median ρ between two labels' maps of the
      same image is 0.087.
    - The data-randomisation test was **not run** (compute).
  - **Faithful: clearly better than random, mostly better than a centre prior.**
    Blur fill, all 485 positive (image, label) pairs:
    - Insertion AUC: 0.740 vs 0.471 (random) and 0.593 (centre). Δ vs centre
      +0.147 [0.134, 0.160], Grad-CAM wins on 88 % of pairs.
    - Deletion AUC: 0.326 vs 0.467 (random) and 0.341 (centre). Δ vs centre
      −0.015 [−0.030, −0.001], paired Wilcoxon p = 0.20, so deletion does not
      separate Grad-CAM from the centre prior.
    - Per label:
      - **DR deletion is worse than centre** (0.331 vs 0.294, Δ +0.037 [0.021, 0.051]).
      - For cataract, deletion does not separate Grad-CAM from random (Δ −0.018,
        p = 0.29). This fits cataract being a global media opacity.
      - AMD insertion ties with centre (Δ −0.040 [−0.080, 0.002]).
    - Mean-colour fill is confounded for cataract: a grey disc looks like an
      opaque lens, so deleting the centre keeps p(cataract) high (centre
      deletion AUC 0.768). Blur is the primary fill.
    - The development checkpoint (trained on these test images) had *not* beaten
      the centre prior on insertion (Δ −0.009 [−0.024, 0.006]). Faithfulness
      depends on the model as well as the method.
  - **DR lesion localisation: above every baseline overall, weak in absolute
    terms, worst for microaneurysms.** 81 IDRiD images, external data; the model
    gives 97.5 % of them p(DR) ≥ 0.5.
    - Any-lesion pixel AUROC is 0.657, against 0.499 (random), 0.575 (centre)
      and 0.518 (wrong-class glaucoma map).
    - EPG is 0.070, 1.7x the lesion area fraction of 0.042.
    - The map's maximum hits a lesion in 11/81 images (0.14), against 0.04
      (random), 0.09 (centre) and 0.10 (glaucoma map).
    - The ±15 px pointing game (0.79) must be read against its chance level: the
      dilated lesions cover 35 % of the FOV.
    - Haemorrhages localise best: AUROC 0.708, pointing game 8/81 (0.10).
    - Hard exudates: AUROC 0.640, *below* the centre map (0.711, Δ −0.071
      [−0.117, −0.027]). Exudates cluster around the macula near the image
      centre.
    - Soft exudates (n = 40): AUROC 0.569, not different from random (Δ +0.067
      [−0.010, 0.139]).
    - **Microaneurysms**: the pointing game hits 1/81, EPG 0.0057 (2.1x the
      0.0027 chance level), AUROC 0.622. Grad-CAM at the 16x16 feature resolution
      (32 px cells at 512 px) cannot point at lesions a few pixels wide, as
      Saporta 2022 found for small pathologies.
    - Localisation correlates weakly with confidence (Spearman ρ between p(DR)
      and EPG = 0.258, p = 0.020).
    - The DR map puts 1.9 % of its mass on the optic disc, against a disc area of
      2.6 %, so there is no optic-disc shortcut on IDRiD. On ODIR, several
      sampled maps peak at the optic disc (DR FP, glaucoma FP, mild-NPDR miss)
      or at the fovea (DR TP).
  - **On-device cost is small on the N100 and moderate on the Pi**, with
    `OPENBLAS_NUM_THREADS=1`:
    - **N100, ORT** (plain 7.5 ms, 108 MB): Grad-CAM for DR adds +1.7 ms
      (9.2 ms) and all 5 labels add +4.9 ms (12.4 ms). Peak RSS +9 / +19 MB.
      The probability-only backbone + NumPy head path costs +0.7 ms.
    - **N100, OpenVINO**: +1.9 / +4.8 ms on 8.2 ms.
    - **Pi 3B, TFLite, 4 threads, 600 MHz cap** (plain 547 ms, 63 MB): +50 ms for
      DR (597 ms, +9 %) and +146 ms for 5 labels (693 ms, +27 %). The NumPy part
      itself is 62 / 156 ms. Peak RSS +7 / +18 MB (70 / 81 MB, well inside the
      870 MB board).
    - **Deployment caveat**: with numpy's default multi-threaded OpenBLAS next to
      the inference runtime, the N100 overhead grew to +79 ms (ORT, 5 labels;
      +9.4 ms with ORT thread spinning off; +18 ms OpenVINO). On the Pi it gave a
      crash and a stall. The NumPy Grad-CAM must be single-threaded
      (`OPENBLAS_NUM_THREADS=1`) when it runs next to an inference runtime.
  - **Paper framing**: Grad-CAM can be offered on both devices at little extra
    cost and is verified to be the reference Grad-CAM. It passes the randomisation
    check, and its relevance ranking is faithful to the model, beating random and,
    on insertion, a centre prior. It is **not** a lesion detector. It overlaps
    haemorrhages above chance, does no better than a centre prior on exudates,
    and does not show microaneurysms, the lesions that define mild NPDR, the
    model's weakest grade. With Sayres 2019 (heatmaps hurt readers on no-DR
    cases) and Ghassemi 2021, the map should be presented as a coarse "where the
    model looked" aid next to a positive prediction, not as evidence that the
    prediction is right.

- **Limitations**:
  - No data-randomisation test. One randomisation seed. One random-map draw per
    image in faithfulness (50 per image in localisation).
  - IDRiD is one Indian clinic and one camera, with the disc truncated at
    top/bottom (black bands after padding). Every image has DR, so the DR map is
    never evaluated on negatives. Masks are binarised with an any-coverage rule
    at 512 px, which slightly enlarges small lesions.
  - Pixel-wise deletion/insertion creates out-of-distribution images (the ROAR
    argument, Hooker 2019). Two fills are reported, with no retraining. Whole
    images are ranked, including the black corners.
  - The figure's operating thresholds (90 % validation sensitivity) are very low
    for H (0.00054) and A (0.039), because pos_weight training compresses those
    probabilities. So the "correct positive" H/A examples are correct only at
    those thresholds.
  - Grad-CAM only (no Grad-CAM++, integrated gradients or Score-CAM), and only
    the last-conv target layer (the only one available on the device without a
    second backbone output).
  - Pi latencies are under the under-voltage cap. Single run per config. The Pi
    clock was sampled before/after each config only.
