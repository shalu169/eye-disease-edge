# Abstract Draft

**Working title:** Affordable Multi-Disease Retinal Screening at the Edge:
Explainable, Uncertainty-Aware Diagnosis on Consumer-Grade Hardware

Diabetic retinopathy, glaucoma, cataract, and age-related macular degeneration
together account for most preventable blindness worldwide, yet screening
remains inaccessible in low-resource regions due to scarce ophthalmologist
coverage and the cost of specialized equipment. Deep learning offers a path
to automated screening, but the literature has three persistent gaps: most
systems target a single disease rather than the multi-morbid reality of
retinal patients; reported accuracy is measured in controlled trials and
degrades sharply in the field (sensitivity as low as 51% has been reported
outside lab conditions); and predictions are delivered as opaque
classifications that non-specialist health workers, who increasingly perform
front-line screening, have no principled basis to trust or override. This
paper presents and evaluates a lightweight multi-label model that jointly
screens for diabetic retinopathy, glaucoma, cataract, and age-related macular
degeneration from a single fundus photograph, entirely on-device, augmented
with calibrated uncertainty estimation and pixel-level Grad-CAM
explainability computed at inference time. We quantize the model to INT8 and
benchmark it across two hardware tiers chosen to reflect real low-resource
budgets: an Intel N100 mini PC (4-core x86, 15 GB RAM, integrated GPU, low-
hundreds-of-dollars) via OpenVINO, and a legacy Raspberry Pi 3B (870 MB RAM,
under-$50) via TensorFlow Lite, reporting per-disease accuracy, inference
latency, peak memory, and power draw on each. We further conduct a field
evaluation with non-ophthalmologist health workers to test whether the
combined confidence-and-explanation output measurably improves triage
decision accuracy and trust relative to a black-box baseline. Results
establish, for the first time, empirical bounds on what multi-disease,
explainable, uncertainty-aware retinal screening is achievable at each
price/resource tier, and the quantized models and benchmarking harness are
released to support reproducible edge-AI ophthalmology research.

---
Word count: ~250. Trim to journal limit (IEEE typically 150-250 words;
Elsevier varies 150-300) once target journal is fixed.
