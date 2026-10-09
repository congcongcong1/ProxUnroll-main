# Nanjing Web-Scene Presentation Guide

Planned speaking time: 600 seconds. Prior decks and reports remain preserved.

## Slide 1

We compare four reconstruction solvers, complete 10000 total HQS updates and evaluate 30 team-curated Nanjing web photographs. These are externally authored photographs, not team captures. Percentage improvement means reduction in equally weighted mean reconstruction MSE, with first-round HQS as the primary reference. Codex assisted with code, experiments, analysis and writing. Architecture and weights: https://github.com/pwangcs/ProxUnroll.

## Slide 2

Y = H X W^T + E simulates integrated single-pixel measurements from existing grayscale images. The experiment does not demonstrate optical acquisition. Source: Duarte et al., doi:10.1109/MSP.2007.914730.

## Slide 3

All four initial solvers use the same matrices and measured values. New and previous HQS variants also share the common MAT operator. DCT-FISTA lambda is fixed from the original separate validation experiment. Sources: Beck and Teboulle, doi:10.1137/080716542; Wang et al., arXiv:2505.23180.

## Slide 4

The supplied restorer uses convolution, window attention and memory between optimization stages within one image. It does not connect video frames. Source diagram and architecture: https://github.com/pwangcs/ProxUnroll.

## Slide 5: Nanjing campus and scenic web photographs

30 location-selected photographs, locked before inference and used only for testing. 18 campus photographs: Gulou and Xianlin, 9 each. 12 scenic photographs: Xuanwu Lake, Sun Yat-sen Mausoleum, Ming Xiaoling and Qinhuai/Fuzimiao, 3 each. These are externally authored web photographs collected by our team, not our own captures. Full-size PNG transcodes via wsrv.nl precede center-square Y grayscale / area resize to 256 x 256. Hashes and dimensions are recorded. This convenience collection contains related sites and repeated photographers.

## Slide 6: Four-method reconstruction on campus scenes

All 18 campus originals, every clean ratio. Four methods share the same common MAT sensing operator and the exact same measurements. No favorable photographs are removed. DCT-FISTA uses the frozen original validation lambda and 200 iterations. Both campus and scenic subset tables remain in the six-page report and comparison/dataset_summary.csv.

## Slide 7: Completed HQS fine-tuning

800 DIV2K training originals and 16 separate selection originals. First 100-step fine-tuning uses official weights. Round 2 initializes its selected weights with a new Adam optimizer; later continuation restores all optimizer, scheduler, RNG and step states. All six sensing factors stay frozen. Direct GT RMSE weights .01 for five intermediate outputs and .95 for final; noise choices 0,0,.01,.05. Data, objective and noise changes are not separately attributed. The same round-2 rule continues from 5000 to 10000 total updates within a 150-minute continuation budget. Exit code 0, actual checkpoint and training audit establish completion. Source: https://data.vision.ee.ethz.ch/cvl/DIV2K/.

## Slide 8: Validation-only checkpoint selection

16 selection originals at sampling 10%/25% and noise 0/.01/.05. Mixed PSNR chooses the eligible best checkpoint, with clean PSNR at least initialization minus .03 dB. Initialization remains a candidate. The 5000-budget checkpoint selected step 4500; continuation selected step 10000. New photographs and their reconstruction scores are not used in selection. Full trace: runs/lab244_10000_20261009/finetune/validation.csv.

## Slide 9: Clean 10% reconstruction error changes

Ratio of equal-photograph mean MSE: 1 - mean candidate MSE / mean reference MSE. Candidate is the validation-selected 10000-step state. Official and 100-step references concern overall fine-tuning changes. The 5000-budget reference is the selected step-4500 state under the same round-2 rule. Noise repetitions are averaged inside each original. These related-site web samples support descriptive conclusions, not population confidence claims. All sources and preprocessing were locked before inference.

## Slide 10: Measurement noise and reconstruction error

All clean/.01/.05 measurement noise levels at 10% sampling, compared with the 100-step HQS checkpoint. The same measurement values reach every method. All three predeclared seeds are averaged per photo; all six locations and every negative case remain in CSV. Strong-noise improvements do not imply equal gains for clean acquisition.

## Slide 11: Case that worsens with continued training

Smallest first-seed MSE reduction relative to the 5000-budget state. Photo web_xuanwu_lake_03, sampling 0.01, noise 0.0. Chosen after testing for diagnosis, not for model selection. Xuanwu Lake: small boats, building boundaries and water ripples blur under 1% clean measurements. Diagnostic skyline rows 106-161 raise MSE 14.02%, versus sky 8.72% and water 5.27%. Those bands are selected after test for diagnosis. Smoothing unsupported fine detail is a plausible prior effect, not an isolated training cause. All negative cases retained. 玄武湖13.jpg; 西安兵马俑; CC BY-SA 4.0; source https://commons.wikimedia.org/wiki/File:%E7%8E%84%E6%AD%A6%E6%B9%9613.jpg; license https://creativecommons.org/licenses/by-sa/4.0/. Adapted by gray center crop, area resize, and (where applicable) reconstruction. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.

## Slide 12: Results and remaining course work

The collection is external-author web photography curated by the team, not self-captured data. One training seed, related scenes and unknown official-pretraining exposure limit generalization. The earlier 32-image DIV2K clean result was 1.84% improvement for the 5000-budget model; it remains preserved. 52 locked DIV2K holdout originals still await evaluation of the 10000-step model. No more training is selected from these test results. Six-page technical report, three-page literature review and this 12-slide deck are available. Timed rehearsal, student understanding and course registration/submission require the students. Codex assisted with code, execution, analysis, writing and slides. Architecture: Wang et al., CVPR 2025.
