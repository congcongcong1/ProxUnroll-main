# Round-2 Presentation Guide

Planned speaking time: 600 seconds. Prior decks and reports remain preserved.

## Slide 1

We compare four reconstruction solvers and extend HQS fine-tuning with a bounded second round. Percentage improvement means reduction in equally weighted mean reconstruction MSE, with first-round HQS as the primary reference. Codex assisted with code, experiments, analysis and writing. Architecture and weights: https://github.com/pwangcs/ProxUnroll.

## Slide 2

Y = H X W^T + E simulates integrated single-pixel measurements from existing grayscale images. The experiment does not demonstrate optical acquisition. Source: Duarte et al., doi:10.1109/MSP.2007.914730.

## Slide 3

All four initial solvers use the same matrices and measured values. New and previous HQS variants also share the common MAT operator. DCT-FISTA lambda is fixed from the original separate validation experiment. Sources: Beck and Teboulle, doi:10.1137/080716542; Wang et al., arXiv:2505.23180.

## Slide 4

The supplied restorer uses convolution, window attention and memory between optimization stages within one image. It does not connect video frames. Source diagram and architecture: https://github.com/pwangcs/ProxUnroll.

## Slide 5: Training, selection and test groups

800 official DIV2K training originals, 16 unchanged validation-selection originals. 32 test originals are fixed evenly indexed samples from the 84 remaining official validation originals; no earlier evaluation or this selection used them. Kodak24 and 24 raw spaced HEVC frames were observed in round 1 and now serve as regression tests. Original pretraining exposure is not fully audited. Source: https://data.vision.ee.ethz.ch/cvl/DIV2K/.

## Slide 6: Initial four-method Kodak evaluation

Preserved round-1 four-method results on the same common MAT operator. Every curve averages all 24 originals. Full CSV: runs/lab220_20261009/baseline/metrics.csv. Higher sampling usually provides more information; this plot does not prove universal recovery.

## Slide 7: Noise-aware fixed-operator fine-tuning

The user authorized 5,000 total updates before any test inference. The initial phase and full-state continuation share the same update rule; validation interval changes to 250. The last recovery state may differ from the validation-selected state. Round 2 starts from first-round selected weights with a new Adam optimizer at 1e-5; later round-2 resume restores full optimizer/RNG/step state. Direct GT weighted RMSE uses .01 for five intermediate outputs and .95 final. Training independently draws sampling ratios and relative measurement RMS 0,0,.01,.05. All six sensing factors stay frozen. The data, loss and noise changes are not separately attributed. Server 244 container luozc_mlvc shares the same workspace with 220.

## Slide 8: Validation-only checkpoint selection

Means use 16 selection originals, 10% and 25% rates, and fixed clean/.01/.05 noise. Combined validation PSNR selects the checkpoint, subject to clean PSNR no worse than initialization minus .03 dB. Initialization participates. No test outcomes choose the checkpoint or stopping. Full trace: finetune/validation.csv.

## Slide 9: Clean 10% reconstruction error changes

MSE reduction = 1 - equally weighted mean new MSE / equally weighted mean reference MSE. These are ratio-of-means percentages, not percentage changes in PSNR or a conversion of mean PSNR gain. Round 1 is the requested primary reference; official HQS is separate. Each HEVC video contributes one equal-weight unit. CSV: comparison/mse_dataset_summary.csv.

## Slide 10: Measurement noise and error changes

All three noise levels at 10%, relative to first-round HQS. Noisy results average three fixed seeds within image. Negative reductions mean MSE increased. Every condition and each whole-video result remain retained. Bootstrap resamples original images or whole videos, but one training seed limits inference.

## Slide 11: Case with the smallest MSE reduction

Post-test diagnostic of the smallest first-seed MSE reduction vs round 1 across all images and predeclared conditions. Image hevc_BasketballDrive_f0167, sampling 0.5, noise 0.0. This case did not choose weights or alter the recipe. All positive and negative image deltas are saved in comparison/mse_paired_deltas.csv.

## Slide 12: Results and remaining limits

One bounded recipe and one training seed. Kodak and HEVC results are repeated regression tests; 32 fresh DIV2K originals were not previously evaluated or used in selection. Original released-model training exposure is not fully audited. Larger training coverage, direct GT loss and noise mixture changed together. Independent HEVC frames do not demonstrate temporal reconstruction. Codex assisted with implementation, execution, analysis, English writing and slides. Sources: https://github.com/pwangcs/ProxUnroll and https://data.vision.ee.ethz.ch/cvl/DIV2K/.
