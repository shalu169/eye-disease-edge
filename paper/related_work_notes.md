# Related-work evidence sheet (BSPC submission)

Generated 2026-10-02. Companion to `refs.bib` (94 entries, all verified).

**How entries were verified.** Bibliographic fields come from the Crossref
`/works/<DOI>` record, with these exceptions: arXiv entries use the arXiv API;
PMLR and NeurIPS entries use the proceedings page meta tags; WHO uses the IRIS
record (handle 10665/328717); software and dataset entries use the live web page
or `CITATION.cff`. Full page ranges for JAMA and JAMA Ophthalmology came from
PubMed `MedlinePgn`, because Crossref only stores the first page for those. The
AAAI page range came from OpenAlex.

**Where the numbers come from.** Every number below was read from the abstract
as returned by Crossref, PubMed (efetch), OpenAlex (abstract_inverted_index) or
Semantic Scholar on 2026-10-02. The source of each abstract is shown in brackets.
"No abstract" means none of these returned one (ScienceDirect blocks automated
access with a captcha). For those entries the only claim allowed is what the
title says. **Read the full text before citing any number not listed here.**

Legend: [CR] Crossref abstract, [PM] PubMed, [OA] OpenAlex, [S2] Semantic Scholar,
[AX] arXiv, [WEB] publisher or project page.

---

## 1. Global burden of vision impairment, the target diseases, and access

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| who2019world | WHO World report on vision | Vision impairment is a global problem and much of it is avoidable | "At present at least 2.2 billion people ... have a vision impairment, of whom at least 1 billion ... could have been prevented or is yet to be addressed" [WEB who.int page description] | WHO IRIS record, ISBN 9789241516570 |
| bourne2021trends | GBD 2019 / VLEG global vision-loss prevalence, 1990-2020, forecast to 2050 | Vision loss is large and growing in absolute numbers | 2020: 43.3M blind, 295M moderate-to-severe VI, 258M mild VI; 2050 forecast: 61.0M blind, 474M MSVI [PM] | DOI 10.1016/S2214-109X(20)30425-3 |
| steinmetz2021causes | GBD 2019 causes of blindness in people aged 50 and over | Cataract, glaucoma, AMD and DR are among the leading causes of blindness, which motivates the label set | Leading causes of blindness in 2020: cataract 15.2M, glaucoma 3.6M, undercorrected refractive error 2.3M, AMD 1.8M, DR 0.86M [PM] | DOI 10.1016/S2214-109X(20)30489-7 |
| burton2021lancet | Lancet Global Health Commission on Global Eye Health | Burden falls mostly on LMICs, and most VI is preventable or treatable | 596M with distance VI in 2020; 90% live in LMICs; >90% have a preventable/treatable cause; productivity loss ~US$410.7 billion PPP/year [OA summary text] | DOI 10.1016/S2214-109X(20)30488-5 |
| teo2021global | Meta-analysis of global DR prevalence projected to 2045 | DR burden | Among people with diabetes: DR 22.27%, VTDR 6.17%. 103.12M adults with DR in 2020, projected 160.50M by 2045 [PM] | DOI 10.1016/j.ophtha.2021.04.027 |
| tham2014global | Meta-analysis of global glaucoma prevalence projected to 2040 | Glaucoma burden | Prevalence 3.54% (age 40-80); 76.0M in 2020, 111.8M in 2040 [PM] | DOI 10.1016/j.ophtha.2014.05.013 |
| wong2014global | Meta-analysis of global AMD prevalence projected to 2040 | AMD burden | Any AMD 8.69% (age 45-85); 196M in 2020, 288M in 2040 [PM] | DOI 10.1016/S2214-109X(13)70145-1 |
| liu2017cataracts | Lancet seminar on cataract | Cataract burden, especially in LMICs | "An estimated 95 million people worldwide are affected by cataract", the leading cause of blindness in middle- and low-income countries [PM] | DOI 10.1016/S0140-6736(17)30544-5 |
| wong2004hypertensive | NEJM review of hypertensive retinopathy | Background on the H label (fundus signs of hypertension) | No abstract (NEJM review). Cite for background only | DOI 10.1056/NEJMra032865 |
| resnikoff2020estimated | ICO survey of ophthalmologist workforce in 194 countries | Specialist scarcity in low-income settings motivates automated screening | 232,866 ophthalmologists in 2015; mean density 3.7 per million in low-income vs 76.2 per million in high-income countries [CR] | DOI 10.1136/bjophthalmol-2019-314336 (online 2019, print vol. 104, 2020) |
| gupta2026early | Narrative review of barriers to glaucoma/DR detection in low-resource settings | Barriers (specialist scarcity, equipment cost, distance) and portable/AI tools as mitigations | No quantitative numbers in the abstract [CR]. This is the PMC13467204 source cited in brainstorm.md. **The "51-86% real-world sensitivity" figure is NOT in its abstract.** The verified primary source for that range is lee2021multicenter | DOI 10.3390/jcm15155827 |

## 2. Seminal deep-learning screening and the real-world performance drop

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| gulshan2016development | Google's DR deep-learning system (JAMA 2016) | Lab-reported referable-DR accuracy is very high | AUC 0.991 (EyePACS-1) and 0.990 (Messidor-2). High-specificity operating point: sensitivity 90.3% / specificity 98.1% (EyePACS-1). Trained on 128,175 images [PM] | DOI 10.1001/jama.2016.17216 |
| ting2017development | Singapore multi-ethnic DLS for referable DR, VTDR, possible glaucoma and AMD | Early multi-disease system, built as separate per-disease models on large private data | Referable DR AUC 0.936; possible glaucoma AUC 0.942; AMD AUC 0.931; external referable-DR AUC 0.889-0.983 [PM] | DOI 10.1001/jama.2017.18152 |
| gargeya2017automated | DR detection with a heatmap | Early DR deep learning that also produced an abnormality heatmap | AUC 0.97 (5-fold CV); 0.94 on MESSIDOR-2 and 0.95 on E-Ophtha [PM] | DOI 10.1016/j.ophtha.2017.02.008 |
| abramoff2018pivotal | Pivotal trial of IDx-DR, the first FDA-authorised autonomous AI | Prospective primary-care performance is lower than retrospective numbers | Sensitivity 87.2%, specificity 90.7%, imageability 96.1%, n=900 [CR] | DOI 10.1038/s41746-018-0040-6 |
| li2018efficacy | Glaucomatous optic neuropathy detection from fundus photos | Single-disease glaucoma deep learning is strong | AUC 0.986, sensitivity 95.6%, specificity 92.0%. False negatives came mainly from coexisting conditions such as high myopia [PM] | DOI 10.1016/j.ophtha.2018.01.023 |
| burlina2017automated | AMD grading from AREDS fundus images | Single-disease AMD deep learning | AUC 0.94-0.96, accuracy 88.4-91.6% [PM] | DOI 10.1001/jamaophthalmol.2017.3782 |
| ruamviboonsuk2019deep | Thai national DR programme, deep learning vs human graders | Deep learning matched or beat graders on a large clinical population | Referable DR sensitivity 0.97 vs 0.74 for graders; specificity 0.96 vs 0.98; n=25,326 images [PM] | DOI 10.1038/s41746-019-0099-8 |
| beede2020human | Human-centred field study of a DR system in 11 Thai clinics | Socio-environmental factors (image quality, workflow, connectivity) degrade deployed performance | Qualitative study; no numbers in the abstract [S2] | DOI 10.1145/3313831.3376718 |
| lee2021multicenter | Head-to-head real-world test of 7 commercial DR algorithms (VA, 311,604 images) | **Primary source for the abstract's "sensitivity as low as 51%" claim** | Sensitivities ranged 50.98-85.90%; NPV 82.72-93.69%; one algorithm had 74.42% sensitivity for PDR [CR/PM]. **Framing note:** the 50.98% is the worst of 7 commercial algorithms on VA teleretinal data. It is not a general "lab-to-field" drop, so word it that way | DOI 10.2337/dc20-1877 |
| ting2019artificial | BJO review of AI in ophthalmology | Overview; names explainability and acceptance of "black-box" AI as deployment challenges | No numbers [CR] | DOI 10.1136/bjophthalmol-2018-313173 |

## 3. Multi-disease / multi-label fundus classification and datasets

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| li2021benchmark | **ODIR dataset paper**: 8 diseases, 10,000 images from both eyes of 5,000 patients | Cite for the ODIR-5K dataset | 5,000 patients / 10,000 images / 8 categories. Found that "simply increasing the scale of network cannot bring good results" [S2] | DOI 10.1007/978-3-030-71058-3_11 (LNCS, Bench 2020 proceedings, 2021). Note: Crossref's PubMed DOI lookup returned an unrelated article; the abstract was taken from S2 and title-matched |
| odir2019challenge | ODIR-2019 challenge page (Peking University) | Dataset provenance (the original source, not the Kaggle mirror) | n/a | [WEB] page live, title confirmed |
| islam2019source | Early 8-class ODIR CNN | Baseline multi-label ODIR performance with a generic CNN | F-score ~85%, Kappa 31%, AUC 80.5% [S2] | DOI 10.1109/SPICSCON48833.2019.9065162 |
| li2020dense | DCNet: uses correlation between paired left/right eyes for ODIR (ISBI) | Patient-level, bilateral multi-label methods | No numbers in the abstract [S2] | DOI 10.1109/ISBI45749.2020.9098340 |
| wang2020multilabel | EfficientNet ensemble on ODIR 2019 | ODIR multi-label work optimises accuracy (ensembles) | No numbers in the abstract [S2] | DOI 10.1109/ACCESS.2020.3040275 |
| pachade2021retinal | RFMiD dataset: 3,200 images, 3 cameras, 46 conditions | Alternative multi-disease dataset, possible external test set | 3,200 images, 46 conditions, adjudicated by 2 retinal experts [CR] | DOI 10.3390/data6020014 |
| cen2021automatic | Multi-label deep-learning platform for 39 fundus conditions | Large-scale multi-disease detection is feasible with big data and big models | 249,620 images; F1 0.923, sensitivity 0.978, specificity 0.996, AUC 0.9984 [CR] | DOI 10.1038/s41467-021-25138-w |
| son2020development | Deep learning for 12 fundus findings, with lesion heatmaps | Multi-finding screening with visual explanation | In-house AUC 96.2-99.9%; external DR-related findings AUC 94.7-98.0% [PM] | DOI 10.1016/j.ophtha.2019.05.029 |

## 4. Lightweight CNNs, quantization and runtimes

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| howard2017mobilenets | MobileNetV1 (depthwise-separable convolutions) | Lightweight CNN lineage | none [AX] | arXiv:1704.04861 |
| sandler2018mobilenetv2 | MobileNetV2 (inverted residuals) | Lineage | (abstract not read, metadata only) | DOI 10.1109/CVPR.2018.00474 |
| howard2019searching | MobileNetV3 (NAS + NetAdapt; Large and Small variants; hard-swish, SE) | Backbone used in this paper | MobileNetV3-Small is "6.6% more accurate compared to a MobileNetV2 model with comparable latency" [OA] | DOI 10.1109/ICCV.2019.00140 |
| hu2018squeeze | Squeeze-and-Excitation blocks | SE blocks inside MobileNetV3, which the TFLite debugger flagged in run 006 | (metadata only) | DOI 10.1109/CVPR.2018.00745 |
| tan2019efficientnet | EfficientNet compound scaling | Alternative backbone (EfficientNet-B0/lite) | B7 84.3% top-1 on ImageNet, 8.4x smaller [AX] | PMLR v97 pp. 6105-6114 (page meta tags), arXiv:1905.11946 |
| deng2009imagenet | ImageNet | Pretraining source | (metadata only) | DOI 10.1109/CVPR.2009.5206848 (Crossref author strings malformed; corrected to "Li, Kai" and "Fei-Fei, Li") |
| wightman2019pytorch | timm library | Model source (`mobilenetv3_small_100`) | n/a | Citation block from the timm README; Zenodo DOI 10.5281/zenodo.4414861 resolves |
| jacob2018quantization | Integer-arithmetic-only inference with QAT (TFLite's scheme) | Basis of INT8 PTQ/QAT; "improvements are significant even on MobileNets" | none [OA] | DOI 10.1109/CVPR.2018.00286 |
| krishnamoorthi2018quantizing | Quantization whitepaper | Per-channel weights + per-layer activations PTQ usually lands within 2% of float; QAT within 1%; 2-3x CPU speedup | numbers as stated [AX] | arXiv:1806.08342 |
| nagel2019datafree | Data-free quantization (cross-layer equalisation + bias correction) | 8-bit PTQ of MobileNets is non-trivial; bias correction helps. Directly relevant to the run-006 TFLite failure (no bias correction) vs NNCF | "frequently leading to either significant performance reduction or engineering time" [OA] | DOI 10.1109/ICCV.2019.00141 |
| nagel2021white | Qualcomm quantization white paper (PTQ vs QAT pipelines) | "In most cases, PTQ is sufficient for 8-bit"; QAT for harder cases | [AX] | arXiv:2106.08295 |
| sheng2018quantization | Quantization-friendly separable convolution | MobileNetV1 quantized models show a "large performance gap" vs float | Modified MobileNetV1 reaches 68.03% top-1 at 8 bits [OA] | DOI 10.1109/EMC2.2018.00011 |
| yun2021mobilenets | Why MobileNets quantize poorly | MobileNets "often have significant accuracy degradation under post-training quantization"; quantization error accumulates more in depthwise-separable CNNs. Supports the run-006 diagnosis ("error spread through the network and accumulates") | [OA] | DOI 10.1109/CVPRW53098.2021.00277 |
| kozlov2021nncf | NNCF (OpenVINO compression framework) | Tool used for N100 INT8 PTQ | [S2] | DOI 10.1007/978-3-030-80129-8_17 |
| onnxruntime | ONNX Runtime | Runtime | n/a | `CITATION.cff` in microsoft/onnxruntime |
| openvino | OpenVINO toolkit docs | Runtime | n/a | [WEB] docs.openvino.ai returns 200. No citable paper found |
| litert | LiteRT / TensorFlow Lite | Pi runtime | n/a | [WEB] ai.google.dev/edge/litert returns 200 |

Not found: a peer-reviewed paper specifically on **hard-swish** PTQ sensitivity.
The "hard-swish/SE are PTQ-sensitive" sentence in EXPERIMENTS.md is therefore
supported only indirectly (yun2021mobilenets, sheng2018quantization, and
nagel2019datafree cover depthwise/MobileNet sensitivity in general). The paper's
own debugger evidence (SE-block MEAN ops flagged) is the direct support.
Phrase it as an observation, not as a known result.

## 5. Edge / on-device / smartphone fundus AI (closest prior work)

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| rajalakshmi2018automated | EyeArt on smartphone-based (Remidio FOP) fundus images, India | Smartphone-camera AI DR screening | Any DR: sensitivity 95.8%, specificity 80.2%; STDR: 99.1% / 80.4%; n=296 [OA] | DOI 10.1038/s41433-018-0064-9 |
| natarajan2019diagnostic | **Medios offline AI on a smartphone**, community screening in Mumbai by minimally trained health workers | Offline on-device DR screening is clinically viable (single disease, proprietary) | RDR sensitivity 100.0% (95% CI 78.2-100), specificity 88.4%; n=213 analysed [OA] | DOI 10.1001/jamaophthalmol.2019.2923 |
| sosale2020simple | SMART study: offline Medios AI on non-mydriatic images | Larger validation of offline smartphone AI | RDR sensitivity 93%, specificity 92.5%; any DR 83.3% / 95.5%; n=900 [CR] | DOI 10.1136/bmjdrc-2019-000892 |
| shen2017portable | Raspberry-Pi-based non-mydriatic fundus camera | Pi-class hardware can serve as the capture device, which motivates Pi-side inference | Components US$185.20, 386 g [CR] | DOI 10.1155/2017/4526243 |
| aljbaar2023dcnn | **Multi-label ODIR models deployed on Jetson Nano** (parallel embedded classifiers, VGG16 and small custom nets) | **Prior multi-disease ODIR + embedded deployment** | Best accuracy 0.9974 / 0.96 (myopia); small net is 20% of VGG16 size [CR]. Mentions Raspberry Pi suitability but implements on Jetson Nano | DOI 10.15587/1729-4061.2023.281790 (low-tier venue) |
| thaware2025leveraging | **8-class multi-label eye-disease framework on Raspberry Pi 5 with TFLite** (EfficientNet-B0 + spatial attention, heatmaps) | **Closest prior work: multi-disease + Raspberry Pi + TFLite + heatmaps already exists** | 96.2% multi-label accuracy; average inference **5 s per image on Raspberry Pi 5**; 13,300 images [OA] | DOI 10.1038/s41598-025-20990-y (the brainstorm's "Sci Rep 2025" item) |
| ajithkumar2025efficient | DenseNet DR grading deployed on Raspberry Pi 4 | Single-disease DR on Pi | 88% accuracy [CR]. No latency in the abstract | DOI 10.11591/eei.v14i2.8248 (low-tier venue) |
| kishore2026edge | CNN DR detection on Raspberry Pi and Jetson Nano (APTOS 2019) | Single-disease DR on Pi/Jetson | 92% accuracy, AUC 0.96 [OA] | DOI 10.1016/j.procs.2026.05.081 (the brainstorm's ScienceDirect S1877050926014171 item). Crossref lists the first author as "S, Kishore", so check the name order on the PDF |
| asare2025deploying | Several CNNs, TFLite INT8, Edge Impulse, on smartphones/MCUs, DR from EyePACS | Single-disease DR with INT8 edge deployment | MobileNet 96.45% accuracy; SqueezeNet 176 KB, 17 ms on GPU [AX] | arXiv:2506.14834 (preprint, not peer-reviewed) |
| abushaban2025optimizing | Pruning, quantization and KD for fundus classification on Jetson Nano | Contrasting result: PTQ was the best efficiency/accuracy trade-off there | 11.22 MB, 97.39% validation accuracy [OA] | DOI 10.1109/ICHORA65333.2025.11017237 |
| nief2025highly | EfficientNet-Lite0 INT8 TFLite for on-device DR | Contrasting claim of "no significant loss" from INT8 on ARM | Up to 91.3% accuracy, <130 ms on ARM mobile, ~4.2 MB [OA] | DOI 10.1109/ICCR67387.2025.11292267 (low-tier) |
| farah2026lightweight | MobileNetV2/EfficientNetB0 for 4-class (cataract, DR, glaucoma, normal) offline Flutter app; compares FP16, DRQ, full-INT8, pruning, KD | **Closest on the quantization comparison**: full-integer INT8 vs FP16 for a retinal MobileNet; FP16 was best | MobileNetV2 91.52%; FP16 4.26 MB at 91.99%; 128-316 ms/image on mid-range phones [OA]. Abstract does not state the INT8 accuracy, so check the PDF | DOI 10.29284/7admmd22 (low-tier) |
| rodriguezcorral2024energy | Energy per image on Edge TPU vs Jetson (Maxwell GPU), glaucoma fundus | Energy benchmarking precedent for fundus edge AI | Classification <10 ms (TPU) / <14 ms (GPU); 45 vs 70 mJ per image [S2] | DOI 10.1016/j.engappai.2023.107298 |
| arnob2025lightweight | Lightweight attention CNN with Grad-CAM/Grad-CAM++, 10-class Bangladeshi fundus data | Explainable "lightweight" retinal CNN without hardware benchmarking (brainstorm's PMC12387214) | 16.6M parameters (not small), 87.9% accuracy, 150x150 px [CR] | DOI 10.3390/jimaging11080275 |
| turker2026knowledge | KD (ConvNeXtV2 to EfficientNet-B0) + Grad-CAM/++ on MultiEYE 9-class | Lightweight + XAI with rigorous protocol (seeds, bootstrap); also evaluates "probabilistic prediction quality" | Macro-F1 0.593 (no KD) to 0.633 (KD) [CR]. Abstract truncated at the results, so check calibration numbers on the PDF | DOI 10.3390/diagnostics16182945 |
| lei2024enhancing | ECVS community-screening pipeline (quality, VI, disease, lesion visualisation) | Closest human-trust prior (brainstorm item arXiv 2410.20309) | AUC 0.98 RETQA, 0.95 PVI, 0.90 EDD; Dice 0.48 VLR [AX]. **Abstract reports no user/trust study** despite "Patient Trust" in the title, so check the PDF before citing it for trust findings | arXiv:2410.20309 (preprint) |
| xu2025edge | Survey of edge deep learning in medical diagnostics | General edge-AI background and hardware taxonomy | none [OA] | DOI 10.1007/s10462-024-11033-5 |

## 6. Uncertainty, calibration and explainability

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| guo2017calibration | Modern NNs are miscalibrated; temperature scaling works | TS + ECE methodology | "temperature scaling ... is surprisingly effective" [AX] | PMLR v70 pp. 1321-1330 (page meta tags) |
| naeini2015obtaining | BBQ calibration (origin of the binned ECE formulation) | ECE definition | none [CR] | DOI 10.1609/aaai.v29i1.9602; pages 2901-2907 from OpenAlex |
| gal2016dropout | MC dropout as approximate Bayesian inference | Alternative UQ (MC dropout) | [AX] | PMLR v48 pp. 1050-1059 |
| lakshminarayanan2017simple | Deep ensembles | Alternative UQ (ensembles), too costly for the Pi | [AX] | NeurIPS 30 proceedings page |
| leibig2017leveraging | MC-dropout uncertainty for DR, uncertainty-informed referral | UQ-based referral improves DR screening | Referring 0-20% most uncertain cases reaches the NHS-recommended 85% sensitivity / 80% specificity [CR] | DOI 10.1038/s41598-017-17876-z |
| ayhan2020expert | Test-time-augmentation uncertainty for DR, validated against experts | Calibrated uncertainty for DR; uncertain cases are hard for physicians too | Qualitative ("well-calibrated") [PM] | DOI 10.1016/j.media.2020.101724 |
| selvaraju2017gradcam | Grad-CAM (ICCV) | XAI method | [OA] | DOI 10.1109/ICCV.2017.74 |
| selvaraju2020gradcam | Grad-CAM (IJCV extended version) | XAI method (cite one or both) | [S2] | DOI 10.1007/s11263-019-01228-7 |
| chattopadhay2018gradcampp | Grad-CAM++ | Variant handling multiple instances | [OA] | DOI 10.1109/WACV.2018.00097 |
| adebayo2018sanity | Sanity checks for saliency maps | Some saliency methods are independent of the model and the data | [AX] | NeurIPS 31 proceedings page |
| arun2021assessing | Trustworthiness of 8 saliency methods on chest X-ray | Saliency maps fail localisation, randomisation and repeatability tests in medical imaging | All 8 failed at least one criterion; pneumothorax AUPRC 0.024-0.224 vs U-Net 0.404 [OA] | DOI 10.1148/ryai.2021200267 |
| saporta2022benchmarking | Benchmark of 7 saliency methods vs a human benchmark on CXR | Grad-CAM is the best saliency method but still worse than humans, especially for small lesions (relevant to microaneurysms) | Gap largest for small, complex-shaped pathologies; confidence correlates with Grad-CAM localisation [CR] | DOI 10.1038/s42256-022-00536-x |
| ghassemi2021false | Lancet Digital Health viewpoint: "false hope" of XAI | Counter-argument; validate rather than rely on explanations | none [OA] | DOI 10.1016/S2589-7500(21)00208-9 |
| sayres2019using | Reader study: DR grades ± integrated-gradient heatmaps for 10 ophthalmologists | **Closest human-factor evidence**: heatmaps help on DR cases but hurt on no-DR cases | Sensitivity for moderate-or-worse DR: unassisted 79.4%, grades only 87.5%, grades + heatmap 88.7%. Heatmaps reduced accuracy for patients without DR (P=0.006) [OA] | DOI 10.1016/j.ophtha.2018.11.016 |

## 7. Label quality, image resolution, and DR datasets

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| krause2018grader | Adjudicated reference standards for DR | **Missed microaneurysms are the top grading error; higher resolution + adjudication raise AUC.** Directly explains the low DR AUC at 224 px with ODIR's patient-level labels | Most common discrepancies: missed MAs 36%, artifacts 20%, misclassified haemorrhages 16%. AUC rose from **0.934 to 0.986** (moderate or worse DR) using adjudicated tuning grades and **higher-resolution input** [PM] | DOI 10.1016/j.ophtha.2018.01.034 |
| sahlsten2019deep | DR/DME grading with higher image resolution | Higher resolution compensates for less training data | Uses "<1/4" of the images "but aided with higher image resolutions" [PM]. No specific resolution numbers in the abstract | DOI 10.1038/s41598-019-47181-w |
| voets2019reproduction | Failed reproduction of Gulshan 2016 on public data | Public-data and single-grade labels lower DR AUC | AUC 0.951 (Kaggle EyePACS) and **0.853 (Messidor-2)** vs the original 0.99 [OA] | DOI 10.1371/journal.pone.0217541 |
| sabottke2020effect | Resolution vs CNN performance (chest X-ray) | General evidence that small-lesion tasks need higher resolution | Max AUC at 256-448 px; nodule AUC at 64 px was 80.7% of the 320 px value [OA]. **Radiography, not fundus**, so cite as analogous evidence | DOI 10.1148/ryai.2019190015 |
| karimi2020deep | Review of label noise in medical deep learning | Label noise is a plausible limit for ODIR's patient-level labels | none [PM] | DOI 10.1016/j.media.2020.101759 |
| cuadros2009eyepacs | EyePACS telemedicine system | EyePACS provenance | No abstract returned | DOI 10.1177/193229680900300315 |
| kaggle2015diabetic | Kaggle Diabetic Retinopathy Detection (EyePACS images) | Dataset | Deadline 2015-07-27 from the Kaggle CLI. EyePACS origin per voets2019reproduction | [WEB] page title and Kaggle CLI |
| aptos2019blindness | Kaggle APTOS 2019 Blindness Detection | Possible DR-head data source | Deadline 2019-09-07 from the Kaggle CLI | [WEB] page title and Kaggle CLI |
| decenciere2014feedback | Messidor database | Dataset | none [CR]. Crossref/OpenAlex give only first page 231 | DOI 10.5566/ias.1155 |
| porwal2018indian | IDRiD dataset (pixel-level DR lesions) | Possible lesion-level test set for Grad-CAM localisation | [CR] | DOI 10.3390/data3030025 |

## 8. Related work in Biomedical Signal Processing and Control (journal fit)

| key | one-line summary | claim it supports | key numbers (source) | verified via |
|---|---|---|---|---|
| gour2021multiclass | Transfer-learning CNNs for ODIR multi-class multi-label | Prior BSPC ODIR work | VGG16 + SGD performed best among 4 backbones [S2]. No metric values in the abstract | DOI 10.1016/j.bspc.2020.102329 |
| he2021multilabel | DCNet patient-level (bilateral) multi-label ocular disease classification | Prior BSPC ODIR-style work (journal version of li2020dense) | "improved with a large margin" [S2], no numbers | DOI 10.1016/j.bspc.2020.102167 |
| he2021self | KD of clinical-feature knowledge into an image-only ocular-disease student | KD for fundus classification in BSPC | no numbers [S2] | DOI 10.1016/j.bspc.2021.102491 |
| sun2022efficientnet | EfficientNet + spatial attention for multi-label fundus disease | BSPC multi-label fundus | **No abstract.** Cite by title only | DOI 10.1016/j.bspc.2022.103768 |
| lu2023automatic | Transfer-learning lightweight CNN for retinal disease classification | BSPC lightweight retinal CNN | **No abstract.** Title only. Check whether any hardware was used | DOI 10.1016/j.bspc.2022.104365 |
| wang2024tlccl | Two-level causal contrastive learning for multi-label ocular disease diagnosis | Recent BSPC multi-label fundus | **No abstract.** Title only | DOI 10.1016/j.bspc.2024.106308 |
| kumar2026enhancing | Optimized MobileNet for DR detection | Recent BSPC MobileNet-based retinal work | **No abstract.** Title only | DOI 10.1016/j.bspc.2025.109207 |

Other BSPC papers found (metadata verified, left out of refs.bib because their
abstracts could not be read):
- 10.1016/j.bspc.2025.108915 (EfficientNetB0 + MobileNetV1 DR grading, 2026)
- 10.1016/j.bspc.2026.111198 (ontology-guided neuro-symbolic explainable ocular diagnosis, 2026)
- 10.1016/j.bspc.2026.111251 (Fed-FTTP federated cross-site multi-label ocular recognition, 2026)
- 10.1016/j.bspc.2026.111493 (uncertainty-based attention in Swin for DR, 2026/27)
- 10.1016/j.bspc.2026.110374 (LAVA-ViT explainable and uncertainty-aware DR, 2026)

The last two are directly on DR uncertainty and explainability, so read them
before submission. Editors may expect them to be cited.

---

## Unverified / dropped

- **Hood & De Moraes 2018, Ophthalmology** (10.1016/j.ophtha.2018.04.020). Dropped: it is a 2-page commentary with the same title as li2018efficacy, not the study.
- **Ouda et al. 2022, Electronics** (10.3390/electronics11131966). Dropped: verified, but the abstract claims 45-disease output with 99% DSC and it is unclear whether ODIR was used. Low value.
- **Wei et al. 2019 SPIE smartphone app**, **Hacisoftaoglu 2020 PRL**, **Jain 2021 IJO**, **Garifulla 2021 Sensors**, **Mehrtash 2020 TMI**: all verified, but dropped to keep the list focused. They can be restored from the DOIs above.
- **Hard-swish-specific PTQ sensitivity paper**: none found. See the note in section 4.
- **OpenVINO peer-reviewed citation**: none found. Cited as a web page.
- **TensorFlow (Abadi et al. OSDI 2016)**: not added because USENIX has no Crossref DOI and it was not checked.
- **Host/organiser attribution for the Kaggle APTOS and DR competitions**: could not be confirmed (pages are JS-rendered), so the author is set to "Kaggle".
- **Numbers for the 5 BSPC title-only entries and full-text claims for farah2026lightweight (INT8 accuracy), turker2026knowledge (calibration results) and lei2024enhancing (trust findings)**: not confirmed.
- **"Sensitivity 51-86% outside controlled trials ... most fundamental limitation" (brainstorm.md §1, attributed to PMC13467204 = gupta2026early)**: not in that paper's abstract. The 50.98-85.90% range is verified in lee2021multicenter and should be cited to that paper.

---

## Gap statement (based only on verified abstracts above)

Deep-learning fundus screening reaches high retrospective accuracy for single
diseases: referable-DR AUC 0.99 \cite{gulshan2016development}, glaucoma AUC
0.986 \cite{li2018efficacy}, AMD AUC 0.94-0.96 \cite{burlina2017automated}.
Large multi-label systems also do well, but they rely on hundreds of thousands
of images and server-class models \cite{ting2017development,cen2021automatic}.
Real-world performance is less reliable: seven commercial DR algorithms ranged
from 50.98% to 85.90% sensitivity on veterans' teleretinal data
\cite{lee2021multicenter}, and clinic conditions limited a deployed system
\cite{beede2020human}. On-device screening has been validated clinically only
for single-disease, proprietary DR software on smartphones
\cite{natarajan2019diagnostic,sosale2020simple}.

Academic edge work either targets DR alone on Raspberry Pi, Jetson or phones
\cite{ajithkumar2025efficient,kishore2026edge,asare2025deploying,nief2025highly},
or runs multi-disease models on more capable boards. Examples are ODIR models on
a Jetson Nano \cite{aljbaar2023dcnn} and an 8-class EfficientNet-B0 on a
Raspberry Pi 5 at about 5 s per image \cite{thaware2025leveraging}. These
abstracts report accuracy and sometimes latency. None of them reports AUC parity
between the training framework and the exported runtime, peak memory, or how
much INT8 post-training quantization costs per disease. Some claim that INT8
costs nothing \cite{nief2025highly}, which contrasts with the known PTQ
sensitivity of MobileNets
\cite{yun2021mobilenets,sheng2018quantization,nagel2019datafree}.

Explainable lightweight retinal models \cite{arnob2025lightweight,turker2026knowledge}
are not benchmarked on low-cost hardware. Calibrated uncertainty has been shown
for DR on workstation models \cite{leibig2017leveraging,ayhan2020expert}, but no
verified edge paper reports calibration (ECE) on-device or checks whether
quantization preserves it.

This paper adds three things:
- a single MobileNetV3-small that jointly screens DR, glaucoma, cataract, AMD
  and hypertensive retinopathy from ODIR-5K;
- a like-for-like benchmark on two price tiers, an x86 N100 mini PC
  (ONNX Runtime/OpenVINO) and a legacy 1 GB-class Raspberry Pi 3B (TFLite).
  It reports per-disease AUC parity with PyTorch, latency, preprocessing cost
  and peak RSS, plus a negative result: naive INT8 PTQ is accuracy-destructive
  for this backbone in TFLite and shifts per-image probabilities enough to
  undermine calibrated uncertainty;
- the methodology to report ECE and Grad-CAM on the edge path.

## Framing implications

1. **Do not claim "first multi-disease screening on Raspberry Pi / edge".**
   thaware2025leveraging (Sci Rep 2025) already runs an 8-class multi-label
   fundus model with heatmaps on a Raspberry Pi 5 via TFLite. aljbaar2023dcnn
   runs ODIR multi-label on a Jetson Nano. Defensible novelty:
   - an older, cheaper tier (Pi 3B, ARMv7, <1 GB RAM) with ~130 ms/image vs
     their ~5 s on a Pi 5 (different models; say so);
   - rigorous export-parity and memory reporting;
   - per-disease INT8 PTQ analysis with diagnosis;
   - calibration-under-quantization.
2. **Fix the abstract's "sensitivity as low as 51% ... outside lab conditions"**:
   cite lee2021multicenter and phrase it as "the least sensitive of seven
   commercial algorithms on real-world teleretinal data (50.98%)".
3. **The INT8 negative result is publishable context.** Some edge-DR papers claim
   negligible INT8 loss (nief2025highly; asare2025deploying "minimal trade-offs"),
   while MobileNet PTQ sensitivity is documented. Run 004/006 adds per-disease
   evidence and the calibration angle.
4. **The DR AUC ceiling (~0.70)** is consistent with krause2018grader (missed
   microaneurysms; resolution and adjudication raised AUC 0.934 to 0.986),
   voets2019reproduction (public, single-grade labels give 0.853 on Messidor-2)
   and sahlsten2019deep (higher resolution helps). Cite these when discussing
   224 px and ODIR's patient-level labels.
5. **Human-factor arm:** sayres2019using showed heatmaps *reduced* accuracy on
   no-DR cases, and ghassemi2021false, arun2021assessing and saporta2022benchmarking
   question saliency reliability (worst for small lesions). If the field study
   stays, pre-register this as a hypothesis to test rather than assuming
   explanations help.
6. **BSPC fit:** cite gour2021multiclass and he2021multilabel (ODIR in BSPC) at
   minimum. Read the 2026 BSPC DR uncertainty/XAI papers listed above.
7. **Methodology risk (not a citation issue):** runs 002-006 pick the best epoch
   on the same val split they report AUC on. The BSPC skill lists this as a
   common rejection reason. Add a held-out test split or CV with CIs.
