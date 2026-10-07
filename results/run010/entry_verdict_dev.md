- **Artifacts**:
  - Scripts: `results/run010_common.py`, `run010_check_reference.py`,
    `run010_export_edge.py`, `run010_export_tflite.py`, `run010_parity_edge.py`,
    `run010_sanity.py`, `run010_train_randlabels.py` (not run),
    `run010_faithfulness.py`, `run010_idrid_prep.py`, `run010_localisation.py`,
    `run010_make_device_fixture.py`, `run010_figures.py`, `run010_tables.py`
    (generates every table in this entry from the JSONs),
    `run010_run_final.sh` (final-checkpoint pipeline). Edge code:
    `edge/gradcam_numpy.py`, `edge/gradcam_bench.py`, `edge/run010_device.sh`.
  - Results (`results/run010/`): `reference_vs_pytorch_grad_cam_dev.json`,
    `parity_{onnx,tflite}_dev007c.json`, `sanity_dev.json` (+`sanity_files_dev.npy`),
    `faithfulness_dev.json`, `faithfulness_dev_per_image.json`,
    `faithfulness_dev_curves.npz`, `localisation_dev.json`,
    `localisation_dev_per_image.csv`, `idrid_manifest.csv`,
    `figure_selection_dev.json`, `figure_idrid_selection_dev.json`,
    `probs_dev_{val,test}.npy`, `tables_dev.md`, `log_*.txt`.
  - Edge artifacts: `edge/artifacts/run010_dev007c_*` and the device bundle
    `edge/artifacts/run010_bundle_dev007c/`.
  - Figures (PDF + PNG): `paper/figs/run010_gradcam_examples`,
    `run010_idrid_overlays`, `run010_deletion_insertion`, `run010_sanity_cascade`.
    Colour map viridis (perceptually uniform). Colour = positive evidence for the
    named label, scaled to the image's maximum: 0 (dark purple) = none, 1 (yellow)
    = strongest. Every figure has a colourbar.
  - IDRiD data: `data/idrid/` (raw zip, sha256, Kaggle metadata, `prep512/`).

- **Verdict (development checkpoint)**:
  - **The edge Grad-CAM is the same Grad-CAM.** The reference matches
    pytorch-grad-cam to 1.2e-6. Over 1500 maps the ONNX/TFLite backbone plus
    the NumPy head gives heatmaps within 1.1e-4 (ONNX) and 1.0e-4 (TFLite) of
    PyTorch autograd (max abs diff on [0,1] maps), probabilities within
    1.5e-5 / 2.4e-5, and an identical argmax in every map. The remaining error
    comes from the backbone runtimes. The NumPy head maths alone agrees to 3e-6
    in the maps and 5e-7 in probability. No autodiff runtime is needed on the
    device.
  - **Passes the model-randomisation sanity check.** Re-initialising only the
    classifier already drops the median Spearman ρ to the original map to 0.08,
    below the 0.11 between maps of two *different* images. At every deeper
    cascade stage it stays at or near that floor (0.01-0.14), so the maps depend
    on the learned parameters. SSIM stays at 0.26-0.42 at every stage, close to
    the different-image level of 0.47. On these smooth, mostly empty maps SSIM is
    dominated by background structure and discriminates poorly, so ρ is the
    informative metric. The maps are class-specific: the median ρ between two
    labels' maps of the same image is 0.155. The data-randomisation test was
    not run (compute; see Methods).
  - **Faithful relative to random, not clearly better than a centre prior.**
    With blur fill, Grad-CAM beats the smooth random map on deletion and
    insertion for every label (all 485 pairs: deletion AUC 0.360 vs 0.482,
    insertion 0.591 vs 0.485, both p < 1e-36). Against the centre-bias map it
    wins on deletion (0.360 vs 0.414, Δ −0.054 [−0.070, −0.036]), but on
    insertion the two are indistinguishable (0.591 vs 0.600, Δ −0.009 [−0.024,
    +0.006]). For cataract the centre map is better on insertion (Δ −0.087). This
    fits cataract being a global media opacity, not a focal finding. The
    mean-colour fill is confounded for cataract: a grey disc looks like an
    opaque lens. Deleting the centre with the mean colour *raises* p(cataract)
    (centre deletion AUC 0.837), and blurring the whole image already gives
    p(cataract) ≈ 0.5. Deletion/insertion is therefore not a valid
    faithfulness test for cataract, and blur fill is the primary setting.
  - **DR lesion localisation: better than every baseline, but weak in absolute
    terms, and worst for microaneurysms.** On 81 IDRiD images (external data;
    the model gives 96 % of them p(DR) ≥ 0.5) the DR map's pixel AUROC for any
    lesion is 0.692, against 0.499 for random, 0.575 for centre and 0.561 for the
    wrong-class glaucoma map. EPG is 0.078, about 1.9x the lesion area (0.042).
    The strict pointing game hits a lesion in 16 % of images (random 4 %, centre
    9 %). The ±15 px pointing game (0.83) is inflated because lesions dilated by
    15 px cover 35 % of the FOV, so it should not be quoted without that chance
    level. By type: haemorrhages AUROC 0.696, PG 0.11; hard exudates 0.708, the
    same as the centre map (0.711, Δ −0.002, n.s.) because exudates cluster
    around the macula near the image centre; soft exudates 0.647. Microaneurysms
    are hardest: the strict pointing game is 0/81 and EPG 0.0055 (2x chance
    0.0027), AUROC 0.650. This matches Saporta 2022: Grad-CAM at 16x16 feature
    resolution (32 px cells at 512 px) cannot resolve lesions a few pixels wide.
    Localisation quality correlates with confidence (Spearman ρ between p(DR)
    and EPG = 0.385, p = 3.8e-4), as Saporta also found. The DR map puts little
    mass on the optic disc (0.8 % vs 2.6 % disc area). On ODIR, though, several
    sampled maps peak at the fovea or the optic disc (example figure), which
    suggests the model partly relies on anatomical landmarks.
  - **Paper framing**: present Grad-CAM as a coarse "where the model looked"
    aid. It is verified to be model-dependent and better than random, and it
    overlaps haemorrhages/exudates above chance. It is **not** a lesion
    detector, and it does not show microaneurysms, the lesions that define mild
    NPDR, the model's weakest grade. Together with Sayres 2019 (heatmaps hurt
    readers on no-DR cases), this argues for showing the map only next to a
    positive prediction, and for not presenting it as evidence of correctness.

- **Limitations**:
  - Dev checkpoint, trained on the test images (faithfulness and example figure
    only; IDRiD is external). The final numbers come from `run010_run_final.sh`.
  - No data-randomisation test. Single random-map draw per image in
    faithfulness (50 per image in localisation). One randomisation seed.
  - IDRiD differs from ODIR: one Indian clinic, one camera, the disc truncated
    top/bottom (black bands after padding), and every image has DR, so the DR
    map is never evaluated on negatives. Masks are binarised with an
    any-coverage rule at 512 px, which slightly enlarges small lesions.
  - Pixel-level perturbation creates out-of-distribution images (Hooker 2019
    ROAR argument). Blur and mean fill are reported, with no retraining.
  - Grad-CAM only (no Grad-CAM++, integrated gradients or Score-CAM
    comparison). One target layer (the only one available on-device without a
    second backbone output).
