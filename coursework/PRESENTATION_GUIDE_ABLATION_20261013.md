# Continuation and Factorial Ablation Presentation Guide

Planned speaking time: 600 seconds. Historical deck and reports remain preserved.

## Slide 1

We simulate single-pixel grayscale reconstruction, compare four solvers, and extend HQS training to 12000 total round-2 updates. Two training seeds each run a matched 2x2 noise/stage-supervision ablation for 800 updates per arm. Architecture and official weights: Wang et al., CVPR 2025, github.com/pwangcs/ProxUnroll. The cover retains the standard scikit-image coins test image from the original project inputs. Codex assisted code, experiments, analysis and materials. Results are condition-specific; no optical camera or temporal video model is claimed. Source evidence: runs/hqs_ablation_20261013/report_facts.json.

## Slide 2

Y = H X W^T + E simulates integrated measurements from existing grayscale images. Fewer measurements leave greater ambiguity. All methods share the same sensing operator and measured values. No physical acquisition is performed. Reference: Duarte et al., DOI 10.1109/MSP.2007.914730.

## Slide 3

Adjoint and DCT-FISTA are classical baselines. FISTA uses 200 iterations and lambda .001 selected only on original independent validation. Released HQS/ADMM share the common MAT operator during evaluation. Architecture: Wang et al., arXiv:2505.23180; FISTA: Beck and Teboulle, DOI 10.1137/080716542. The newly implemented work is matched evaluation and controlled fine-tuning rather than a new network architecture.

## Slide 4

The source diagram describes the supplied convolution/attention restorer and stage feature memory. Source: Wang et al., CVPR 2025, github.com/pwangcs/ProxUnroll. Memory connects stages within one image, not previous/future video frames.

## Slide 5

Training uses 800 DIV2K originals and validation uses 16 distinct originals. The primary test is the previously locked, project-unused 52 DIV2K originals. Regression tests include 110 previously observed images: Kodak24, eight HEVC videos with three spaced frames each, DIV2K32, and 30 Nanjing web photographs. The photographs shown are external-author regression images, not team captures. Source manifests and original licenses are preserved. DIV2K: data.vision.ee.ethz.ch/cvl/DIV2K/.

Photograph attribution:
Zhongshan Building in Nanjing University Gulou Campus 01 2012-11.JPG; 猫猫的日记本; CC BY-SA 3.0; source https://commons.wikimedia.org/wiki/File:Zhongshan_Building_in_Nanjing_University_Gulou_Campus_01_2012-11.JPG; license https://creativecommons.org/licenses/by-sa/3.0/. Adapted by grayscale center crop and area resize. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.
南大仙林天文台.jpg; panoramio用户KrUDSO4†www.panoramio.com; CC BY-SA 3.0; source https://commons.wikimedia.org/wiki/File:%E5%8D%97%E5%A4%A7%E4%BB%99%E6%9E%97%E5%A4%A9%E6%96%87%E5%8F%B0.jpg; license https://creativecommons.org/licenses/by-sa/3.0/. Adapted by grayscale center crop and area resize. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.
Night View of Qinhuai River with Scenic Cruise in Fuzi Miao 20170519.jpg; 4084470 0.smil; CC BY 4.0; source https://commons.wikimedia.org/wiki/File:Night_View_of_Qinhuai_River_with_Scenic_Cruise_in_Fuzi_Miao_20170519.jpg; license https://creativecommons.org/licenses/by/4.0/. Adapted by grayscale center crop and area resize. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.

## Slide 6

This preserved graph contains four-method results on all 18 campus originals, at five clean ratios, under identical measurements. At clean 10%, official HQS/ADMM have mean PSNR 28.35/28.38 dB, compared with 24.60 for DCT-FISTA and 23.92 for adjoint. These are previously observed regression results, not the new primary52 experiment. Source: runs/nanjing_web30_20261009/comparison/dataset_summary.csv.

## Slide 6

This preserved graph contains four-method results on all 18 campus originals, at five clean ratios, under identical measurements. At clean 10%, official HQS/ADMM have mean PSNR 28.35/28.38 dB, compared with 24.60 for DCT-FISTA and 23.92 for adjoint. These are previously observed regression results, not the new primary52 experiment. Source: runs/nanjing_web30_20261009/comparison/dataset_summary.csv.

## Slide 7

All four arms start from identical official HQS weights with fresh Adam 1e-5, batch one, FP32, frozen sensing factors, 800 DIV2K training and 16 validation originals. Seeds 20261013 and 20261014 each complete 800 steps per arm. Same-seed image order, crops, flips, sampling rates and potential noise draws are identical. Removing noise sets amplitude to zero. Removing intermediate loss keeps total weight one. Separate continuation restores full step-10000 state and adds 2000 updates under the unchanged update rule. Source: runs/hqs_ablation_20261013/lock.json and all training/pairing audits.

## Slide 8

The validation-selected continuation is step 11500; all 12000 round-2 updates completed. Selection uses mixed validation PSNR with clean initialization minus .03 dB guard. No test score selects weights. Primary52 was locked before continuation and was not used in project training or selection; original official-pretraining exposure is unknown. Bars are equal-original mean-MSE reductions at 10% sampling; repeats are averaged inside each original. Sources: evaluation_primary/paired_summary.csv, continuation/audit.json and evaluation_lock.json under runs/hqs_ablation_20261013.

## Slide 9

Fixed-budget endpoint comparison: noise-on versus noise-off with weighted stage supervision, same official starting state, seed, image/crop/sampling sequence and 800 updates. Each bar is one training seed on the primary52 set at 10% sampling. Noise levels are test measurement noise, not training-seed labels. Positive means adding training noise reduces MSE. All seeds and conditions, including negative values, are retained. Validation-selected comparisons are separately available. Source: evaluation_primary/paired_summary.csv and seed_effect_stability.csv.

## Slide 10

Fixed-budget endpoint comparison: weighted stage loss versus final-only loss under noisy training. Five intermediate weights .01 plus final .95 are compared with final-only weight 1, preserving total loss weight. All other factors, source order, crops, sampling and measurement noise are paired within each training seed. Primary52 test, 10% sampling. Positive means adding intermediate supervision reduces MSE. The full 2x2 design also estimates effects under the other factor level and their interaction. Two seeds do not establish robust significance. Source: evaluation_primary/paired_summary.csv and factorial_effects.csv.

## Slide 11

Diagnostic image holdout10000_div2k_0897; original source /datasets/DIV2K_valid_HR/0897.png; sampling 0.04, noise 0.0, measurement seed 2026. Reference hqs_10000; candidate hqs_12000; MSE reduction -4.9656%, PSNR delta -0.2105 dB. Selected after testing for diagnosis only. The selected continuation is used unless it never regresses, in which case the endpoint is explicitly labeled. Absolute error from clipped floating predictions, scale 0 to .3. The error pattern locates mismatch, not a proven training cause. DIV2K source dataset: data.vision.ee.ethz.ch/cvl/DIV2K/. Full negative CSV remains preserved.

## Slide 12

New formal updates: 8400. Primary52 mean clean/strong-noise MSE changes versus prior selected HQS: 0.2084% / 0.0897%. The selected and fixed-budget views are distinct. Four ablation arms, two training seeds and full paired provenance are complete; the factors concern 800-step adaptation from official weights and do not isolate the entire historical 10000-step recipe. Primary and regression evaluations retain 11440 and 14520 rows, with floating metric verification. Software grayscale simulation, no temporal video inputs, unknown upstream pretraining exposure. Nanjing photographs are external-author web images. Codex assisted execution, analysis and writing. Students must understand the code and rehearse; course-platform submission is not performed.
