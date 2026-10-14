# Final Presentation Guide (2026-10-14)

Planned speaking time: 600 seconds. This twelve-slide version includes the completed 20000-update evaluation. Historical versions remain preserved.

## Slide 1 (20 seconds)

We simulate grayscale single-pixel reconstruction, compare four methods, fine-tune official HQS with frozen sensing factors, and run paired noise/stage-loss ablations with two training seeds. The main continuation completes 20000 total updates. Validation selects step 19500. The 20000 endpoint stays separate. On primary52, the extension slightly increases MSE relative to the prior selected step 11500, while SSIM rises slightly. Architecture and weights belong to Wang et al., CVPR 2025, github.com/pwangcs/ProxUnroll. The cover retains the scikit-image coins example. Codex assisted code, experiments, analysis and materials. Sources: technical_report_flow_20261014.pdf and both frozen experiment roots.

## Slide 2 (40 seconds)

Y = H X W^T + E simulates integrated measurements from existing grayscale images. Fewer measurements leave greater ambiguity. All methods share the same sensing operator and measured values. No physical acquisition is performed. Reference: Duarte et al., DOI 10.1109/MSP.2007.914730.

## Slide 3 (45 seconds)

Adjoint and DCT-FISTA are classical baselines. FISTA uses 200 iterations and lambda .001 selected only on original independent validation. Released HQS/ADMM share the common MAT operator during evaluation. Architecture: Wang et al., arXiv:2505.23180; FISTA: Beck and Teboulle, DOI 10.1137/080716542. The newly implemented work is matched evaluation and controlled fine-tuning rather than a new network architecture.

## Slide 4 (45 seconds)

The source diagram describes the supplied convolution/attention restorer and stage feature memory. Source: Wang et al., CVPR 2025, github.com/pwangcs/ProxUnroll. Memory connects stages within one image, not previous/future video frames.

## Slide 5 (55 seconds)

Training uses 800 DIV2K originals and validation uses 16 distinct originals. The 52 DIV2K originals were project-unused when locked for the previous experiment. The 20000 extension now reuses them and the 110 previously evaluated regression originals. This is a continuation check on observed test inputs. Test scores do not select weights or change the recipe. Original-source splits remain disjoint. Official pretraining exposure is unknown. Regression tests include 110 previously observed images: Kodak24, eight HEVC videos with three spaced frames each, DIV2K32, and 30 Nanjing web photographs. The photographs shown are external-author regression images, not team captures. Source manifests and original licenses are preserved. DIV2K: data.vision.ee.ethz.ch/cvl/DIV2K/.

Photograph attribution:
Zhongshan Building in Nanjing University Gulou Campus 01 2012-11.JPG; 猫猫的日记本; CC BY-SA 3.0; source https://commons.wikimedia.org/wiki/File:Zhongshan_Building_in_Nanjing_University_Gulou_Campus_01_2012-11.JPG; license https://creativecommons.org/licenses/by-sa/3.0/. Adapted by grayscale center crop and area resize. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.
南大仙林天文台.jpg; panoramio用户KrUDSO4†www.panoramio.com; CC BY-SA 3.0; source https://commons.wikimedia.org/wiki/File:%E5%8D%97%E5%A4%A7%E4%BB%99%E6%9E%97%E5%A4%A9%E6%96%87%E5%8F%B0.jpg; license https://creativecommons.org/licenses/by-sa/3.0/. Adapted by grayscale center crop and area resize. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.
Night View of Qinhuai River with Scenic Cruise in Fuzi Miao 20170519.jpg; 4084470 0.smil; CC BY 4.0; source https://commons.wikimedia.org/wiki/File:Night_View_of_Qinhuai_River_with_Scenic_Cruise_in_Fuzi_Miao_20170519.jpg; license https://creativecommons.org/licenses/by/4.0/. Adapted by grayscale center crop and area resize. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.

## Slide 6 (55 seconds)

This preserved graph contains four-method results on all 18 campus originals, at five clean ratios, under identical measurements. At clean 10%, official HQS/ADMM have mean PSNR 28.35/28.38 dB, compared with 24.60 for DCT-FISTA and 23.92 for adjoint. These are previously observed regression results, not the new primary52 experiment. Source: runs/nanjing_web30_20261009/comparison/dataset_summary.csv.

## Slide 7 (60 seconds)

All four arms start from identical official HQS weights with fresh Adam 1e-5, batch one, FP32, frozen sensing factors, 800 DIV2K training and 16 validation originals. Seeds 20261013 and 20261014 each complete 800 steps per arm. Same-seed image order, crops, flips, sampling rates and potential noise draws are identical. Removing noise sets amplitude to zero. Removing intermediate loss keeps total weight one. Separate continuation restores full step-10000 state and adds 2000 updates under the unchanged update rule. Source: runs/hqs_ablation_20261013/lock.json and all training/pairing audits. A further authorized continuation restores the full 12000-update state, including Adam, scheduler, RNG and selection history, and adds 8000 unchanged-recipe updates to 20000. This takes 270.32 minutes and peaks at 10.84 GiB. The ablations retain their original 800-step budgets. The two continuations add 10000 updates, and eight ablation runs add 6400. Sources: runs/hqs_20000_20261013/finetune/{config,summary,audit}.json.

## Slide 8 (75 seconds)

All three chart series share the selected 10000-step reference. At clean/noise .05, selected step 11500 reduces MSE by 0.2084% / 0.0897%. Selected step 19500 instead gives -0.6207% / -0.3172%. A negative MSE reduction means error rises. Relative to step 11500, selected 19500 increases MSE by 0.8309% / 0.4073%. PSNR deltas are -0.0107 / -0.0019 dB, and SSIM deltas are 0.000364 / 0.000800. The endpoint stays separate and also regresses on these primary MSE conditions. Initialization is an eligible validation candidate. No test result selects either state. These inputs were evaluated before this extension. Against official HQS, selected 19500 still reduces MSE by 3.3191% / 11.9766%, which describes accumulated adaptation, not the incremental benefit of the last 8000 updates. Sources: runs/hqs_20000_eval_20261014/evaluation_primary/paired_summary.csv and the frozen previous paired_summary.csv.

## Slide 9 (60 seconds)

Fixed-budget endpoint comparison: noise-on versus noise-off with weighted stage supervision, same official starting state, seed, image/crop/sampling sequence and 800 updates. Each bar is one training seed on the primary52 set at 10% sampling. Noise levels are test measurement noise, not training-seed labels. Positive means adding training noise reduces MSE. All seeds and conditions, including negative values, are retained. Validation-selected comparisons are separately available. Source: evaluation_primary/paired_summary.csv and seed_effect_stability.csv.

## Slide 10 (55 seconds)

Fixed-budget endpoint comparison: weighted stage loss versus final-only loss under noisy training. Five intermediate weights .01 plus final .95 are compared with final-only weight 1, preserving total loss weight. All other factors, source order, crops, sampling and measurement noise are paired within each training seed. Primary52 test, 10% sampling. Positive means adding intermediate supervision reduces MSE. The full 2x2 design also estimates effects under the other factor level and their interaction. Two seeds do not establish robust significance. Source: evaluation_primary/paired_summary.csv and factorial_effects.csv.

## Slide 11 (50 seconds)

Diagnostic image holdout10000_div2k_0897; original source /datasets/DIV2K_valid_HR/0897.png; sampling 0.04, noise 0.0, measurement seed 2026. Reference hqs_10000; candidate hqs_12000; MSE reduction -4.9656%, PSNR delta -0.2105 dB. Selected after testing for diagnosis only. The selected continuation is used unless it never regresses, in which case the endpoint is explicitly labeled. Absolute error from clipped floating predictions, scale 0 to .3. The error pattern locates mismatch, not a proven training cause. DIV2K source dataset: data.vision.ee.ethz.ch/cvl/DIV2K/. Full negative CSV remains preserved. The displayed illustration remains the validation-selected 11500 checkpoint, internally named hqs_12000. It is not a 19500 reconstruction. The 20000-budget selected state has 258/572 primary measurement conditions with higher MSE than step 11500. All negative cases and per-image deltas remain available. Source: runs/hqs_20000_eval_20261014/evaluation_primary/negative_cases.csv and paired_image_deltas.csv.

## Slide 12 (40 seconds)

The selected 19500 checkpoint shows a small primary-MSE regression against selected 11500 and a small SSIM increase. Regression dataset changes are mixed and mostly fractions of a percent. Positive table values indicate lower MSE. Kodak remains separate from training and validation. Each HEVC video contributes equally, using three fixed spaced raw frames. We do not claim a further universal 5–10% improvement. The full comparison saves 3564 new final floating reconstructions and 21384 stage metrics over 162 originals. All new final metrics are independently recomputed, with identical per-condition measurements and source/operator hashes. Frozen official/10000/11500 scores pass a fresh reference-equivalence smoke check before reuse. The 800-step two-seed ablations remain distinct from the 20000 continuation. Sources: runs/hqs_20000_eval_20261014/{results,completion_audit}.json and evaluation_regression/paired_summary.csv. Limitations: modest tests, two ablation seeds, upstream pretraining exposure unknown, external-author photographs, grayscale software acquisition. Students must understand the code, rehearse and submit the coursework themselves.
