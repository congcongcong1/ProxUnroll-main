# Topic 7: Single-Pixel Imaging and Compressed Sensing

## Team

| Name | English name used in reports | Student ID |
| --- | --- | --- |
| 罗子聪 | Zicong Luo | 522026230043 |
| 羊书浩 | Shuhao Yang | 502026230083 |
| 栗宇博文 | Yubowen Li | 502026230092 |

## Deliverables

The latest completed extension is the verified 5000-step experiment, with
`output/pdf/technical_report_5000_20261009.pdf` and
`output/presentation/course_presentation_5000_20261009.pptx`.
The original deliverables listed below remain preserved. Teacher-requirement status
is recorded in [REQUIREMENTS_STATUS_20261009.md](coursework/REQUIREMENTS_STATUS_20261009.md).
The GitHub evidence snapshot, external data/checkpoint locations, and offline checks
are documented in [REPOSITORY_SYNC.md](coursework/REPOSITORY_SYNC.md).

- `output/pdf/technical_report.pdf`: English technical report in a two-column conference style.
- `output/pdf/literature_review.pdf`: separate English comparative literature review.
- `output/presentation/course_presentation.pptx`: editable presentation with speaker notes.
- `coursework/PRESENTATION_GUIDE.md`: presentation timing, explanation guide, and questions.
- `coursework/results/`: raw experiments, validation, summaries, provenance, and reconstructions.
- `coursework/figures/`: scientific plots generated from those results.

The original study uses the released pretrained ProxUnroll models. The laboratory
extension additionally fine-tunes HQS with frozen sensing factors. It does not claim
to reproduce every CVPR benchmark result or build a physical single-pixel camera.
The evaluation includes classical comparisons, noise experiments, and failure analysis.

## Reproduce the Experiments

Run from the repository root. Python 3.12 is the verified version.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-coursework.txt
.venv/bin/python -m coursework.prepare_data
.venv/bin/python -m pytest tests/test_coursework.py -q
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv/bin/python -m coursework.run_experiments --device cpu --output coursework/results-repeat
```

The delivered results remain preserved. The runner requires a fresh output directory
and rejects overwriting an existing `metrics.csv`. A quick execution uses `--smoke`
and another output directory; smoke results are not the report experiment.
The report and summarizer read the delivered `coursework/results` directory.

```bash
MPLCONFIGDIR=/tmp/proxunroll-matplotlib .venv/bin/python -m coursework.summarize
.venv/bin/python -m coursework.build_reports
.venv/bin/python -m coursework.verify_results \
  --source-snapshot coursework/archive_original/source \
  --output tmp/original-verification.json
```

The PPTX itself is editable without Codex. Its generator uses the Codex bundled
`@oai/artifact-tool` runtime and presentation finalizer. To rebuild inside Codex,
run `python -m coursework.prepare_presentation_asset`, then
`node coursework/build_presentation.mjs`. Set `PRESENTATION_FILENAME` to a fresh
filename for a revision. `CODEX_RUNTIME_ROOT` and `PRESENTATIONS_SKILL_DIR` can
override the discovered local runtime and skill locations. The finalizer requires
a new output path and checks the native chart data before delivery.
Each build uses a fresh directory under `tmp/presentation/` for its candidate,
validation receipt, and slide previews, so revisions do not collide with prior checks.

The pinned requirements describe the verified macOS CPU environment. CPU execution
does not require CUDA. Other operating systems or future package indexes may require
compatible wheel builds; retain the original provenance when comparing environments.
The original-result audit uses the preserved source snapshot because later laboratory
extensions changed the runner; it writes a fresh audit under `tmp/`.
The separate upstream training script additionally needs albumentations and
scikit-learn. Those training dependencies and training data are not needed for this
coursework's evaluation commands.

## Experimental Contract

1. Six regular evaluation images: camera, astronaut, coins, moon, page, clock.
2. Two disjoint validation images: coffee and chelsea. Three DCT-FISTA lambda values,
   three clean rates, one globally selected value, 200 iterations.
3. Two procedural stress targets: checkerboard and small_text. They are not
   self-collected photographs and are excluded from regular-image means.
4. All images are 256 x 256 grayscale. RGB sources become OpenCV Y before resizing.
   Area resizing may change aspect ratio. This is not Set11 or CBSD68.
5. Four methods: adjoint, DCT-FISTA, pretrained HQS, pretrained ADMM.
6. Five clean nominal rates: 1%, 4%, 10%, 25%, 50%. Actual ratios use ceiling-rounded
   row counts capped by the stored matrix size. Nominal 50% uses 181 x 181
   measurements (49.9893%), because only 181 rows are available in each factor.
7. At 10% nominal sampling, measurement-relative RMS Gaussian noise uses sigma 0.01
   and 0.05, each with three seeded realizations and identical inputs across solvers.
8. Main results use `measurement_matrix/blind_learned_256_256_matrices.mat` for every
   method. The matrices inside the two checkpoints differ. This means the main
   comparison includes transfer to a common operator. `native_operators.csv` separately
   evaluates each checkpoint's own operator at 10%, together with a matched baseline.
9. PSNR and skimage SSIM operate on clipped [0,1] outputs, without border removal.
   Metrics are computed before 8-bit PNG export. Noise repeats are averaged within
   each image before image-level means. Stress results remain separate.
10. Neural stage results and disabling HQS memory are inference interventions, not
    retrained architecture or loss-function ablations.

## Reading the Code

- `model/proxunroll.py:ProxUnroll.reconstruct`: measurement-only inference. Its default
  factors preserve original checkpoint behavior; external factors enable fair tests.
- `coursework/baselines.py`: separable forward/adjoint operators and DCT-FISTA.
- `coursework/run_experiments.py`: strict checkpoint load, validation, timing,
  shared-input evaluation, intermediate stages, and native-operator diagnostics.
- `coursework/summarize.py`: aggregation, paired image bootstrap, and figures.
- `coursework/build_reports.py`: complete editable report text and PDF generation.

Do not interpret `prox_g` in the upstream training forward path as an inference-time
source of ground truth. The new reconstruction function never takes a reference image.
The upstream color evaluator retains reference chroma, so this project evaluates
grayscale only and makes no full-color recovery claim.

## Source and Data Attribution

- Original code and weights: https://github.com/pwangcs/ProxUnroll
- Wang et al., CVPR 2025: https://arxiv.org/abs/2505.23180
- Duarte et al., IEEE SPM 2008: https://doi.org/10.1109/MSP.2007.914730
- Author-hosted classic paper: https://mdav.ece.gatech.edu/publications/ddtlskb-spm-2008.pdf
- Beck and Teboulle, FISTA: https://doi.org/10.1137/080716542
- Data descriptions and credits: https://scikit-image.org/docs/stable/api/skimage.data.html

The supplied upstream `LICENSE` is preserved. The local starting directory was a
source archive with no `.git` metadata, so no original source commit is claimed.
Checkpoint, source, sensing-matrix, and processed-image hashes are recorded for this run.
The figures under `fig/` come from the upstream repository; newly measured figures
live under `coursework/figures/`.

## Before Submission

- Confirm the English spellings of names and any teacher-specific cover/template rules.
- Each member should explain the measurement equation, the adjoint, DCT sparsity,
  the HQS/ADMM loop, and why common sensing factors matter.
- Review the actual failure images and the limits of the small test suite.
- Keep the AI disclosure in the report and presentation. Do not claim unperformed
  retraining, self-capture, hardware experiments, or individual contributions.
- The Git repository is local. Publishing to a school platform or a remote repository
  is a separate action and has not been performed.

## AI Assistance

Codex assisted with implementation, experiment execution, analysis, English writing,
and presentation preparation. The students are responsible for understanding,
reviewing, and presenting the submitted work. The ProxUnroll architecture and
pretrained weights are attributed to the original authors.

## Laboratory Extension (2026-10-09)

The server-220 extension uses a separate Linux/CUDA environment, external data
manifests, fresh output directories, and bounded fixed-operator HQS fine-tuning.
See `coursework/LAB220.md` for the exact protocol, commands, checkpoints, and
measured outcome. The original experiment and deliverables above remain preserved.
The revision uses `technical_report_lab220_20261009.pdf` and
`course_presentation_lab220_20261009.pptx` after completed evidence verification.
Historical source files corresponding to the original experiment hashes are in
`coursework/archive_original/source`. Verify that original evidence with:

```bash
.venv/bin/python -m coursework.verify_results \
  --source-snapshot coursework/archive_original/source \
  --output runs/lab220_20261009/original-verification.json
```

New external results use `coursework.verify_lab_run`, which recomputes the metrics
from retained float reconstructions and checks shared-measurement and file hashes.

### 5000 步续训（2026-10-09）

用户追加授权第二轮总计 5000 步，MSE 相对第一轮 100 步 HQS 下降 5–10% 为目标。
实际使用经 upload 核验的 244 服务器 / luozc_mlvc，共享项目根目录不变。
独立运行 `runs/lab244_5000_20261009`，从前阶段第 946 步恢复完整训练状态；测量矩阵继续冻结。
运行方式、预算扩展与文件索引见 [ROUND5000.md](coursework/ROUND5000.md) 和 [EXTEND5000_PROTOCOL.md](coursework/EXTEND5000_PROTOCOL.md)。
实际已完成 5000 步，验证选中第 4500 步；两段正式训练共 121.8 分钟，峰值分配显存 10.84 GiB。
相对上一轮 100 步模型，新增 DIV2K 32 张在干净 10% 采样下平均 MSE 下降 1.84%（95% 配对区间 0.68–3.48%），未达到最低 5%；强噪声 .05 下下降 11.89%。
Kodak/HEVC 是回归测试，干净条件提升也有限；各视频和所有退步条件均保留。
三种 HQS 权重的 2640 条指标和新增四方法的 1408 条指标均从浮点重建独立重算，最大误差 0。
原报告、第一轮报告、PPT 和所有权重保留，新增材料使用 5000 日期后缀。
新版英文技术报告六页：`output/pdf/technical_report_5000_20261009.pdf`；PPT 十二页：`output/presentation/course_presentation_5000_20261009.pptx`。
中文答辩补充见 [DEFENSE_5000_ZH.md](coursework/DEFENSE_5000_ZH.md)，逐页讲稿见 [PRESENTATION_GUIDE_5000.md](coursework/PRESENTATION_GUIDE_5000.md)。

### 总计 10000 步的续训启动（2026-10-09）

用户授权进一步训练后，已于北京时间 13:16 在 244 / luozc_mlvc 的空闲 GPU5 启动独立续训。
目录 `runs/lab244_10000_20261009`，从完整第 5000 步状态继续到总计最多 10000 步（新增 5000）。
保持梯度规则、数据、损失、学习率、验证选择和冻结矩阵不变；真实恢复一致性与首个第 5001 步 checkpoint 已验证。
预计约两小时，正式续训上限 150 分钟；启动和回访命令见 [ROUND10000.md](coursework/ROUND10000.md)。
预先固定剩余 52 张新测试原图，旧 80 张明确为回归测试。本次不自动测试；用户回来后核对完成状态，再公平评测。
现有报告/PPT 仍是已完成的 5000 步结果；不宣称尚未评测的续训有提升。
