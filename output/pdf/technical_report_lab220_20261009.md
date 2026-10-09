# Single-Pixel Imaging under Limited Measurements:<br/>External Evaluation and Fixed-Operator Fine-Tuning

Zicong Luo (522026230043), Shuhao Yang (502026230083), Yubowen Li (502026230092)

## Abstract

We extend a reproducible single-pixel imaging simulation to 24 public Kodak images and 24 raw frames from eight laboratory-held HEVC sequences. Four reconstruction methods receive identical measurements. We then fine-tune the released HQS reconstruction network for 100 steps with fixed sensing factors, using 128 DIV2K training images and 16 disjoint validation images. Kodak and all HEVC videos remain outside training and model selection. We retain raw metrics, floating-point reconstructions, provenance and complete recovery state. The measured test changes are reported with negative cases and their limits.

## 1. Scope and Related Work

Single-pixel imaging obtains integrated responses under known masks. Duarte et al. [1] connect this acquisition to compressed sensing. Our software starts with existing images, creates measurements numerically, and compares reconstruction methods. It does not demonstrate optical acquisition or fabricate a self-captured dataset.

Wang et al. [2] release HQS and ADMM ProxUnroll variants. A shared image restorer combines convolution, window attention and stage memory. Their proximal trajectory training motivates the bounded fine-tuning experiment here. Beck and Teboulle [3] motivate our DCT-FISTA baseline. We attribute the architecture and official weights to their authors.

## 2. Measurement and Reconstruction

The separable forward model is Y = H X W<super>T</super> + E. X is a 256 x 256 grayscale image. H and W are cropped rows of the released MAT-file factors. All solvers use the same Y, H and W. The adjoint is H<super>T</super> Y W. DCT-FISTA minimizes 0.5 ||H X W<super>T</super> - Y||<sub>F</sub><super>2</super> + lambda ||DCT(X)||<sub>1</sub> with 200 iterations and lambda = 0.001, frozen from the original disjoint validation study.

HQS alternates residual correction with learned restoration. ADMM also updates a dual variable. The measurement-only reconstruct API accepts no reference image, and regression tests match the original clean forward computation. The final prediction follows six restorer stages. Clean-image targets appear only in training labels and metric computation.

The common factors differ from factors embedded in the released checkpoints. Comparisons therefore include transfer to a common sensing operator. This is documented rather than attributed entirely to architectural superiority. The original experiment also retained native-operator diagnostics.

## 3. Data and Evaluation Protocol

Kodak24 is a public still-image benchmark. HEVC B contributes BQTerrace, BasketballDrive, Cactus, Kimono1 and ParkScene. HEVC E contributes FourPeople, Johnny and KristenAndSara. Each entire sequence belongs to the test split. We select zero-based frames 0, floor(N/3) and floor(2N/3) before reconstruction, giving 24 spaced frames.

Raw HEVC files match the canonical resolutions, integer byte lengths for 8-bit planar 4:2:0. BQTerrace contains 601 frames and BasketballDrive 501, each one more than its nominal version. We record and use the actual count. We read the original Y plane directly. Raw files carry no self-describing color-range metadata. We preserve stored uint8 Y codes divided by 255, record observed extrema and percentiles, and apply no inferred limited/full-range expansion. RGB PNG images use OpenCV RGB2YCrCb Y.

Every test and validation image receives the same spatial rule: maximal centered square crop, then OpenCV INTER_AREA resize to 256 x 256. This preserves aspect ratio but excludes side content. Manifests retain source paths, original dimensions, source hashes, raw frame hashes, frame indices, crop coordinates and processed PNG hashes. Results apply to these processed views.

Clean nominal ratios are 1%, 4%, 10%, 25% and 50%. Ceiling-rounded factor row counts determine actual ratios. At nominal 50%, stored factors cap the measurement count at 181 x 181, or 49.9893%. At 10%, Gaussian noise has RMS scale 0.01 or 0.05 relative to clean Y, with seeds 2026, 2027 and 2028.

Predictions are clipped to [0,1] before PSNR and skimage SSIM, with data_range = 1 and no border removal. Metrics precede PNG quantization. Noise repeats average within image, frames average within video, and video means average equally within each HEVC class. Kodak averages equally across original images. We report classes and videos separately.

## 3.1. Preserved Initial Study

The original six regular test images, two disjoint validation images and two procedural stress targets remain intact. That study retained 352 main metric rows and 1,056 neural stage rows, including sampling, noise, stage interventions and failure cases. Its 6-page technical report, 3-page reading report and 12-slide presentation remain as prior versions. This revision uses fresh output directories.

## 3.2. Remote Reproducibility

We verified the SSH peer as 192.168.1.220, inspected luozc_nvenc_new and its /workspace and /datasets bind mounts, and matched synchronized file hashes. A project-local Linux virtual environment contains PyTorch 2.6.0 with CUDA 12.4. The selected GPU is a 24 GiB RTX 3090, identified by UUID. No container-wide package installation or destructive synchronization is used.

![Figure 1. Public Kodak24 quality under the common operator. Noise repeats average within each image.](../../coursework/figures/external_quality.png)

## 4. External Test Results

Table 1 reports clean 10% PSNR in dB. HEVC class results give equal weight to each video. Full sampling/noise results, SSIM and timings remain in CSV rather than a selected-image score.

Dataset | Adjoint | FISTA | HQS | ADMM

Kodak | 24.67 | 25.18 | 29.38 | 29.43

HEVC_B | 25.88 | 26.77 | 31.22 | 31.28

HEVC_E | 24.90 | 26.30 | 33.55 | 33.51

The complete four-method extension contains 2,112 metric rows and 6,336 stage rows. Every condition has one shared measurement hash across all solvers. Verification recomputed metrics from 2,112 retained float arrays. This establishes consistency with stored reconstruction evidence.

The sweep shows how information loss and measurement noise affect each solver. A larger test set improves coverage, but it cannot establish universal recovery or deployment latency. CPU DCT-FISTA and GPU neural timings represent different execution backends. They must not be presented as a hardware-matched speed comparison.

The finite six-stage trajectories and memory interventions from the original study remain inference experiments. This extension does not retrain an architecture ablation or prove asymptotic convergence. Video frames are reconstructed independently, with no temporal input or reference propagation.

## 4.1. Results by Video

Table 2 reports 10% clean results for every selected video. Each row averages its three preselected, spaced frames. Five B videos and three E videos are separate experimental units.

Sequence | HQS dB | ADMM dB

BQTerrace | 29.53 | 29.61

BasketballDrive | 32.79 | 32.84

Cactus | 29.10 | 29.18

Kimono1 | 33.73 | 33.79

ParkScene | 30.93 | 30.98

FourPeople | 33.23 | 33.25

Johnny | 36.44 | 36.46

KristenAndSara | 31.00 | 30.81

## 5. Fixed-Operator Fine-Tuning

A fixed seeded selection takes 128 original images from DIV2K train [4]. Sixteen evenly indexed DIV2K validation originals provide selection. We separate source images before any crop and check source hashes across splits. Kodak and all eight HEVC videos are absent from fine-tuning and selection. Original pretraining exposure has not been fully audited, so untouched here refers to this new training procedure.

Each training step samples a valid random square whose side lies between 256 and the smaller source dimension, then applies area resizing and seeded flips. Step-keyed image order and crop randomness survive resume. This avoids oversized random crops in the upstream training loader. We use batch size one and clean ratios cycling over 1%, 4%, 10%, 25% and 50%.

Training starts from the official HQS state. Its 256 x 256 factors are set to the common MAT operator, and all six sensing-factor parameters are frozen. The optimizer learns reconstruction parameters only. We retain the upstream weighted proximal trajectory RMSE, with weights 0.01 for five intermediate stages and 0.95 for the final stage. Adam uses a fixed 5e-06 learning rate and gradient-norm clipping at 1.

Initialization loads model weights strictly and creates a new optimizer. Full resume additionally restores Adam, scheduler, RNG states and the optimization step, with matching data/operator/configuration contracts. Atomic checkpoint writes are reopened and checked. The selected best state and a complete last recovery state are separate artifacts.

![Figure 2. Validation selects the checkpoint. Test error bars bootstrap original images or whole videos, with one training seed.](../../coursework/figures/training_test.png)

## 5.1. Cost and Held-Out Comparison

The 3-step smoke verified finite loss, unchanged sensing factors and loadable optimizer checkpoints. Its mean step cost was 1.82 s with peak allocated GPU memory 10.85 GiB. The continuation updated 96 more steps in 173.1 s, reaching 100 total steps in 221.4 s across the training chain. It selected step 100, and reached 10.85 GiB peak allocated memory.

Table 3 gives the selected HQS checkpoint minus original HQS at 10% sampling. The same test images, operator, noise and metric code apply to both. The last training checkpoint is also evaluated, even when validation selects an earlier state.

Dataset | Delta dB | Delta SSIM | 95% interval

Kodak | +0.135 | +0.0028 | [+0.095, +0.184]

HEVC_B | +0.146 | +0.0018 | [+0.118, +0.171]

HEVC_E | +0.372 | +0.0027 | [+0.253, +0.549]

At relative noise RMS 0.05, selected-model PSNR changes are Kodak +0.003 dB, HEVC B -0.031 dB and HEVC E -0.041 dB. SSIM increases by about 0.003-0.004. Thus clean-measurement gains coexist with weak or negative PSNR changes under stronger noise. The fine-tuning data include only clean measurements.

Intervals come from 5,000 paired bootstrap samples at the dataset unit level. They describe this one run and small dataset, with only five B and three E sequences. They cannot quantify variability across independent training seeds. We report 25% and noisy 10% comparisons separately in the raw comparison tables.

The measured clean 10% changes are Kodak: +0.135 dB; HEVC_B: +0.146 dB; HEVC_E: +0.372 dB. A validation gain does not guarantee a test gain. Every image delta remains in paired_comparison.json, including regressions. We identify the worst changes for inspection and preserve their reconstructions. No additional ADMM training or repeated tuning against Kodak/HEVC results is performed in this first bounded extension.

![Figure 3. Smallest first-seed test change: kodim20, 10% sampling, noise RMS 0.05, delta -0.248 dB. Diagnostic chosen after evaluation, never used for selection.](../../coursework/figures/regression_case.png)

## 6. Limitations and Conclusions

This project compares four methods on shared synthetic measurements and extends coverage to public Kodak images and laboratory-held HEVC content. It provides a bounded, fixed-operator fine-tuning experiment whose benefit is determined by the measured untouched-test comparison, rather than training loss or selected photographs.

Important limits remain: centered crop/resizing changes the benchmark task, the raw-video range has no embedded provenance, the original pretrained-data exposure is not fully audited, and the common operator differs from checkpoint-native factors. We use one fine-tuning seed and a small image subset. The test comparison is conditional on these preprocessing and measurement rules.

The simulation omits optical calibration, mask constraints, detector noise models, quantization and photon statistics. No physical single-pixel camera or full-color recovery is demonstrated. Each HEVC frame is an independent grayscale inverse problem. The model does not use previous or future frames, so these results do not establish a video reconstruction network.

## 7. Evidence and Reproduction

Remote root: /workspace/ProxUnroll-main. Run root: runs/lab220_20261009. data/ holds manifests and preprocessing records. baseline/ and comparison/ hold raw CSV, stage CSV, float arrays, PNGs and provenance. finetune/ holds train/validation CSV, best.pth, last.pth and the bounded-run summary. logs/ holds detached-job logs, PID records and exit codes. environment.txt freezes Linux package versions.

Reproduce data with python -m coursework.prepare_lab_data --output NEW_DIR. Evaluate with python -m coursework.run_experiments --manifest NEW_DIR/test.json --output NEW_RESULTS --device cuda. Train with python -m coursework.finetune --manifest NEW_DIR/train_val.json --output NEW_TRAIN --max-steps 100 --max-minutes 10. Use the CUDA/BLAS exports in the guide. Use --init only for new weight initialization, or --resume for full state recovery. The project guide records exact measured-run commands and checkpoint hashes.

## AI Disclosure

Codex assisted with implementation, remote execution, analysis, English writing and presentation preparation. The team must review and understand the evidence. The architecture and official weights belong to the cited authors. No student-specific contribution or course-platform submission is claimed.

## References

[1] M. F. Duarte et al. Single-Pixel Imaging via Compressive Sampling. IEEE Signal Processing Magazine 25(2), 83-91, 2008. doi:10.1109/MSP.2007.914730.

[2] P. Wang et al. Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging. CVPR, 2025. arXiv:2505.23180. github.com/pwangcs/ProxUnroll.

[3] A. Beck and M. Teboulle. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM J. Imaging Sciences 2(1), 183-202, 2009. doi:10.1137/080716542.

[4] E. Agustsson and R. Timofte. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. CVPR Workshops, 2017. DIV2K: data.vision.ee.ethz.ch/cvl/DIV2K/.
