"""Write a Chinese run guide and defense notes using verified measurements."""
import json
from coursework.run_experiments import ROOT


def main():
    base=ROOT/'runs/nanjing_web30_20261009'
    facts=json.loads((base/'report_facts.json').read_text())
    analysis=facts['analysis'];train=facts['training'];lock=json.loads((base/'data/selection_lock.json').read_text())
    def value(d,s=0,ref='hqs_round1'):
        return next(r for r in analysis['datasets'] if r['dataset']==d and r['cr']==.1 and r['sigma']==s and r['reference']==ref and r['candidate']=='hqs_10000')
    rows=['| 数据 | 对官方 HQS | 对 100 步 HQS | 对 5000 预算 HQS |','|---|---:|---:|---:|']
    noise=['| 数据 | 干净 | 噪声 .01 | 噪声 .05 |','|---|---:|---:|---:|']
    for d,label in [('NJU_campus_web','校园 18 张'),('Nanjing_scenic_web','景区 12 张')]:
        rows.append('| '+label+' | '+' | '.join(f'{value(d,ref=r)["mse_reduction_percent"]:+.2f}%' for r in ['hqs','hqs_round1','hqs_5000'])+' |')
        noise.append('| '+label+' | '+' | '.join(f'{value(d,s)["mse_reduction_percent"]:+.2f}%' for s in [0,.01,.05])+' |')
    locations=['| 地点 | 原图数 | 干净 10% MSE 下降 | 噪声 .05 MSE 下降 |','|---|---:|---:|---:|']
    for loc in ['NJU_Gulou','NJU_Xianlin','Xuanwu_Lake','Sun_Yat_sen_Mausoleum','Ming_Xiaoling','Qinhuai_Fuzimiao']:
        records=[next(r for r in analysis['locations'] if r['location']==loc and r['cr']==.1 and r['sigma']==s and r['reference']=='hqs_round1' and r['candidate']=='hqs_10000') for s in [0,.05]]
        locations.append(f'| {loc} | {records[0]["images"]} | {records[0]["mse_reduction_percent"]:+.2f}% | {records[1]["mse_reduction_percent"]:+.2f}% |')
    worst=facts['worst_case'];negative=analysis['negative_10000_conditions_by_reference']
    text=f'''# 南京校园与景区网络图片测试（2026-10-09）

## 数据是什么

本组自行整理了 30 张网络照片：南京大学鼓楼、仙林校区各 9 张；玄武湖、中山陵、明孝陵和秦淮河/夫子庙各 3 张。
包含建筑、树木、水面、文字、航拍和夜景。它们是外部作者拍摄的照片，不是我们亲自拍摄。
报告使用 **team-curated web photographs**，中文使用 **自行整理的南京校园与景区网络图像集**。
老师鼓励的手机自采仍是可选、未完成的加分项；增加网络场景不能替代拍摄事实。

来源按地点和场景在重建前固定，没有按模型成绩挑图。全部作者、拍摄日期、许可和链接在
`runs/nanjing_web30_20261009/ATTRIBUTION.md` 与 `coursework/nanjing_web30_sources.json`。
存在同一地点的相近视角和重复摄影作者；不把这 30 张当成随机抽样或独立的大型基准。

直连 Wikimedia 图像超时，改由 wsrv.nl 获取全尺寸 PNG 转码，未请求缩放、裁剪或滤镜。
保留了尺寸、来源 URL 和 PNG 哈希。它们不是原 JPEG 字节，解码、色彩配置或 EXIF 处理可能与直读原文件不同。
`source_png/` 保存下载的全尺寸 PNG（本地保留，不进入 Git）；`data/` 保存实际评测的 30 张灰度图。

统一预处理：EXIF 方向 → 最大中心正方形 → OpenCV RGB2YCrCb 的 Y → INTER_AREA 256×256 → uint8/255。
清单记录每张图的原尺寸、裁剪坐标、原作者、来源、全尺寸 PNG 和灰度图哈希。
已在任何推理前检查整张联系表、源图/处理图精确重复和感知近似哈希；所有 30 张保留。

## 测试和真实训练状态

数据只用于测试，不参与 DIV2K 800 张训练原图、16 张验证原图的训练或选模。
验证选择的权重、数据清单、预处理和测量矩阵均在首次推理前锁定于 `data/selection_lock.json`。
公开照片是否出现在官方原始预训练中尚未全面审计，不能说绝对没有任何预训练暴露。

7 个方法/状态：伴随重建、DCT-FISTA、官方 HQS、官方 ADMM、100 步 HQS、5000 预算 HQS（验证选中 4500 步）、10000 步 HQS。
五个干净采样率 1/4/10/25/50%；10% 采样再加相对测量 RMS 噪声 .01/.05，各重复三个固定种子。
相同 MAT 测量矩阵、测量值、噪声实现和 PSNR/SSIM 规则；DCT-FISTA 使用原先独立验证选定的 lambda .001，固定 200 次迭代。
所有主指标先由浮点重建计算，显示 PNG 的量化不参与主指标。视频扩展仍是逐张重建，不是时序模型。

10000 步续训已经实际结束，退出码 0：新增 5000 步耗时 {train['wall_seconds']/60:.2f} 分钟，所有第二轮阶段累计 {facts['total_training_wall_seconds']/60:.2f} 分钟。
峰值分配显存 {train['peak_memory_mib']/1024:.2f} GiB；验证选中第 {train['best_step']} 步。
训练审计确认 10000 次更新、Adam 步数、有限损失/梯度、冻结的六个测量因子和完整恢复状态。
没有根据这 30 张的结果另选 checkpoint 或继续无预算训练；原先锁定的 DIV2K52 尚未评测，留待下一次回访。

## 实际效果

下表为干净 10% 采样，10000 步权重相对各参考的 **平均 MSE 下降百分比**。正数更好，负数更差。
先在每张原图内平均噪声重复，再等权平均原图 MSE，最后取均值之比；不是 PSNR 数值的百分比。

{chr(10).join(rows)}

10% 采样，对 100 步权重，各噪声条件分别报告：

{chr(10).join(noise)}

每个地点的结果，对 100 步权重：

{chr(10).join(locations)}

全部 330 个条件中，10000 步比官方/100 步/5000 预算模型更差的条件分别为
{negative['hqs']}/{negative['hqs_round1']}/{negative['hqs_5000']}，全部保留在 `comparison/negative_cases.csv`。
最差第一种子续训退步示例：`{worst['image']}`，采样率 {worst['cr']}，噪声 {worst['sigma']}，对 5000 预算模型 MSE 下降 {float(worst['mse_reduction_percent']):+.2f}%（负值即误差增加）。
报告展示同输入的两版重建及误差图。高频细节被平滑是可能解释，未经消融证实，不能写成确定因果。
之前 DIV2K32 的 5000 预算模型干净 MSE 仅下降 1.84%，原结论保留，不能用新网络图片改写它。

完整推理保存 **2310 条主指标、9900 条阶段指标、330 个配对条件**。
独立核验重新生成测量哈希，并从全部 2310 个浮点 NPY 重算 PSNR/SSIM，最大差异 0。
数据、来源代码、矩阵、权重哈希均通过。`verification.json` 保存核验记录。

## 目录和重跑命令

经核验的服务器：`192.168.1.244`；容器：`luozc_mlvc`；根目录：`/workspace/ProxUnroll-main`。
它与 220 的项目/数据挂载共享存储，服务器身份分开检查。实验使用空闲 GPU5 RTX 3090 的 UUID，不影响其他已有任务。
项目独立环境 `.venv-linux`，Python 3.11.11 / Torch 2.6.0+cu124；评测完整包版本在 `comparison/provenance.json`。

- `runs/nanjing_web30_20261009/data/`：精确灰度输入、来源清单、协议、人工检查与选模锁。
- `runs/nanjing_web30_20261009/smoke/`：单图最小验证，7 条主指标，浮点重算通过。
- `runs/nanjing_web30_20261009/comparison/`：所有 CSV、汇总、负例、PNG、浮点 NPY、版本和哈希。
- `runs/nanjing_web30_20261009/logs/web30_comparison.log`、`.exit`：推理/复核和首次汇总的日志。原退出码 1 是汇总 JSON 保存的数据类型错误；推理 COMPLETE 和 2310 个浮点复核已通过。
- `runs/nanjing_web30_20261009/logs/analysis_finalize.log`、`.exit`：修复后单独补跑汇总，实际退出 0；原失败日志保留。
- `runs/nanjing_web30_20261009/sync_receipt.json`：40 个文件的无删除同步和逐文件哈希对比。
- `runs/lab244_10000_20261009/finetune/best.pth`：第 10000 步验证选择的推理权重。
- `runs/lab244_10000_20261009/finetune/last.pth`：第 10000 步完整训练恢复状态。
- `runs/lab244_10000_20261009/logs/finetune_10000.log`、`training_audit.json`：训练和审计记录。

以下命令在容器项目根目录执行。输出目录必须改成新目录，不能覆盖交付结果。

```bash
export CUDA_VISIBLE_DEVICES=GPU-16f6ac58-4d8c-8845-0886-1dc146a0327e
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv-linux/bin/python -m coursework.evaluate_manifest \\
  --manifest runs/nanjing_web30_20261009/data/test.json \\
  --data-dir runs/nanjing_web30_20261009/data \\
  --output runs/NEW_WEB30/comparison \\
  --methods adjoint,fista_dct,hqs,admm,hqs_round1,hqs_5000,hqs_10000 \\
  --extra-checkpoint hqs_round1=runs/lab220_20261009/finetune/best.pth \\
  --extra-checkpoint hqs_5000=runs/lab244_5000_20261009/finetune/best.pth \\
  --extra-checkpoint hqs_10000=runs/lab244_10000_20261009/finetune/best.pth
.venv-linux/bin/python -m coursework.verify_lab_run runs/NEW_WEB30/comparison
.venv-linux/bin/python -m coursework.analyze_nanjing_web30 runs/NEW_WEB30/comparison
```

本次由 `coursework.launch_lab_job` 使用 nohup 启动，并用 timeout 1800 将推理/核验总预算限制为 30 分钟。
SSH 断开不影响作业。启动、GPU 进程、完成标记、退出码和原始重建均有实际检查记录。
取回结果：

```bash
.venv/bin/python -m coursework.fetch_lab_run --server 244 \\
  --run runs/nanjing_web30_20261009 --parts comparison,logs
```

默认不取回 NPY；要在本地重算全部浮点指标，增加 `--include-arrays`，并准备锁定的三个微调权重。
微调权重和 NPY 保留在远端，也已取回本地核验；全尺寸来源 PNG 本地保留。Git 只收录选定冻结证据和实际评测的灰度输入。
所有照片及其图像派生版本保留来源许可和署名，代码许可证不覆盖照片。

新版英文六页报告 `output/pdf/technical_report_nanjing_20261009.pdf`、十二页 PPT `output/presentation/course_presentation_nanjing_20261009.pptx`。
旧版文件不变。三页文献报告保持；新增讲稿计划仍为 600 秒。课程登记、实际排练、理解代码和最终课程平台提交未由本次操作代办。
'''
    local=json.loads((base/'comparison/local_float_verification.json').read_text())
    text+=f'''\n## 跨平台核验补充

全部 {local['float_arrays_recomputed']} 个重建 NPY 已取回本地，独立重算指标，最大差异 {local['max_metric_error']:.3e}。
Mac ARM 与记录的 Linux x86 重新生成测量值时，{local['local_regeneration_bitwise_different_conditions']}/330 条件的末位比特不同；
最大绝对差 {local['local_regeneration_max_absolute_difference']:.3e}、相对 L2 差 {local['local_regeneration_max_relative_l2_difference']:.3e}。
正式实验内所有方法始终共享同一测量值。`audit/measurements.npz` 重新存档远端的 330 个测量数组，逐个匹配原 CSV 测量哈希。
本地依据该存档核验，不把 CPU 库的末位差当成数据变更；NPZ 和 NPY 均保留在远端及本地，不进入 Git。
`comparison/local_float_verification.json` 保存每个重建数组哈希、包版本及上述差异。

```bash
.venv/bin/python -m coursework.verify_nanjing_floats \\
  runs/nanjing_web30_20261009/comparison \\
  --measurement-archive runs/nanjing_web30_20261009/audit/measurements.npz
```
'''
    (ROOT/'coursework/NANJING_WEB30.md').write_text(text)
    defense=f'''# 南京场景与 10000 步实验答辩补充

1. **这些图片是自己拍的吗？** 不是。我们自行选定、整理和预处理了外部作者的网络照片，保留了来源和许可。自采拍摄属于老师鼓励的可选项，未因此完成。
2. **项目真正完成了什么？** 在电脑上生成 Y=HXW^T+E，用同测量比较四种重建方法，再比较三种微调状态；没有制作光学相机，也没有使用视频前后帧。
3. **训练真的跑了吗？** 实际更新 10000 步，Adam 步数一致，退出 0，checkpoint 和日志可查；不是只启动或只看下降的训练损失。
4. **矩阵会不会跟着训练使比较不公平？** 模型里矩阵是参数，但本次六个 H/W 张量全部冻结；评测各模型一律覆盖使用同一个 MAT 算子和相同测量值。
5. **怎样判断改善？** 平均 MSE 的比值，分校园/景区、地点、采样率和噪声；正值才表示更好。不是 PSNR 数字涨 5%。
6. **为什么看 5000 与 10000 的差？** 两者使用同一第二轮更新规则、完整状态续训，能比较增加训练步数后的验证选中模型。100 步到第二轮还同时改变数据覆盖、损失、噪声，不能把效果全归因于步数。
7. **有无挑好的图？** 清单和 checkpoint 哈希在测试前固定；全部 30 张、11 条件、三个噪声种子都保留。所有退步条件也进入 CSV。
8. **失败图怎么解释？** `{worst['image']}` 在相同输入下续训退步 {abs(float(worst['mse_reduction_percent'])):.2f}% MSE；误差图定位细节偏差。自然图像先验平滑高频是解释假设，没有消融就不证明因果。
9. **还有哪些局限？** 一次训练种子，小规模相关网络场景，中心裁剪灰度任务，官方预训练暴露未知。DIV2K52 锁定测试仍待跑。不能推广为所有相机或所有视频都改善。
10. **哪些要本人完成？** 理解测量算子、HQS/FISTA、冻结和恢复区别；三人分工，实际计时排练 10 分钟；课程登记、上传和 2 分钟答问。

AI 声明：Codex 协助代码、实验执行、分析和材料；模型架构、官方权重和图片作者都另行署名。学生需要核查并理解。
'''
    (ROOT/'coursework/DEFENSE_NANJING_ZH.md').write_text(defense)


if __name__=='__main__':main()
