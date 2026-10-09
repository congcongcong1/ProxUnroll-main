# Lab220 Presentation Guide

Planned speaking time: 600 seconds. Original slides and reports remain preserved.

## Slide 1

We compare four solvers on public Kodak24 and spaced frames from eight laboratory-held HEVC videos. We fine-tune official HQS for 100 bounded steps. Raw measurements and all test deltas are retained. Codex assisted with code, experiments, analysis and writing. Model source: https://github.com/pwangcs/ProxUnroll.

## Slide 2

Y = H X W^T + E is a simulation from an existing grayscale image. A detector obtains one integrated response per pattern. Multiple patterns constrain the inverse problem. Reference: Duarte et al., doi:10.1109/MSP.2007.914730.

## Slide 3

Adjoint, DCT-FISTA, official HQS and official ADMM share the same measurements. The HQS fine-tuned variant also uses the same fixed MAT operator. DCT regularization comes from the original disjoint validation study. References: Beck and Teboulle, doi:10.1137/080716542; Wang et al., arXiv:2505.23180.

## Slide 4

This architecture comes from the released ProxUnroll work. Stage memory joins adjacent optimization stages within a single image. It does not join video frames. Source image and architecture: https://github.com/pwangcs/ProxUnroll.

## Slide 5: External evaluation protocol

24 Kodak images and three zero-based spaced frames from each of five HEVC B and three HEVC E sequences. The rule is 0, floor(N/3), floor(2N/3), using actual stored lengths. BQTerrace has 601 frames and BasketballDrive 501. We use original raw Y planes, maximal center-square cropping and area resize. Raw Y codes are divided by 255 without range expansion. All are public or laboratory-held data. Sources and SHA-256 are in data/test.json.

## Slide 6: Kodak reconstruction across sampling rates

Clean Kodak means over 24 original images. All solvers use the common MAT operator. Nominal 50% caps at 181 × 181 measurements or 49.9893%. Full CSV and independently recomputed float-array metrics: baseline/metrics.csv and baseline/verification.json.

## Slide 7: Results by dataset at 10% sampling

Kodak treats each original image as one unit. HEVC B and E first average three frames within each video, then average videos equally. There are five B and three E independent videos. All video results remain in baseline/video_summary.csv. Neural timings use CUDA and DCT-FISTA uses CPU, so timing is not a hardware-matched comparison.

## Slide 8: Bounded HQS fine-tuning

128 seeded DIV2K training originals and 16 disjoint DIV2K validation originals. Kodak and whole HEVC videos are untouched by this new training and selection. Original pretraining exposure is not fully audited. All six sensing factors are frozen. Training uses the upstream proximal trajectory weighted RMSE and a fixed 5e-6 Adam learning rate. Full optimizer, scheduler, RNG and step state are saved separately from selected best weights. DIV2K source: https://data.vision.ee.ethz.ch/cvl/DIV2K/.

## Slide 9: Validation selects the checkpoint

Validation averages 16 originals at clean 10% and 25% rates. Model selection never reads test images. Best step: 100. Bounded training stops at the recorded step/time limit. Full trace: finetune/validation.csv. The short run measured 1.82 seconds per step. Resume verification is recorded separately.

## Slide 10: Test change after HQS fine-tuning

Selected checkpoint minus original HQS under shared measurements. Both 10% and 25% clean rates are reported. Noise results, last-checkpoint results and every image delta are preserved in comparison/paired_comparison.json. Bootstrap uses original Kodak images or whole HEVC videos. A single training seed and few videos limit inference. Positive validation change alone is insufficient evidence of test gain.

## Slide 11: A difficult held-out test case

Outcome-based diagnostic only. This image has the smallest selected-model minus original change among the preselected first-seed conditions across all test images. It was not used for selecting the checkpoint or changing the training recipe. Image: kodim20. Sampling: 10%. Noise RMS: 0.05. Delta PSNR: -0.2477 dB. Every positive and negative delta is preserved. Figure source: retained raw test reconstructions.

## Slide 12: Conclusions and limits

We completed remote sync, larger paired evaluation and bounded fixed-operator HQS fine-tuning. Test changes are conditional on this crop/resizing rule and common operator. Original pretraining exposure is not fully audited. One training seed does not establish a general fine-tuning benefit. HEVC frames are independent images, with no previous/future-frame input. No optical camera, full-color system or course-platform submission is claimed. Codex assisted with code, execution, analysis, writing and presentation. Architecture and weights: Wang et al., CVPR 2025, https://github.com/pwangcs/ProxUnroll.
