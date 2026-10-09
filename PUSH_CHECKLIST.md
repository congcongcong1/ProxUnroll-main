# 仓库同步与交付清单

本次用户已授权同步至 https://github.com/congcongcong1/ProxUnroll-main。
2026-10-09 实际核对：仓库是 Public，默认分支 master；本地 origin 指向同一仓库。
同步前本地与远端均为 a51da7c；最新提交以 Git 历史与远端核验为准。

## 本次版本

- 原有代码、官方权重、测量矩阵、数据和报告保留。
- 收录 Linux/CUDA 环境要求、实验室数据清单评测、固定矩阵微调、完整状态恢复验证和运行协议。
- 收录已完成的 100 步与 5000 步实验 CSV、图表、来源/权重哈希、审计及英文报告/PPT。
- 收录 10000 步续训的固定方案和启动证据；没有把启动写成完成或测试提升。
- 仓库证据索引、外部实验文件位置与检查命令见 `coursework/REPOSITORY_SYNC.md`。

## 同步与核验

普通 Git 同步使用：

```bash
git fetch origin master
git status --short --branch
git log --oneline origin/master..HEAD
git push -u origin master
git rev-parse HEAD
git ls-remote --heads origin master
```

两处 SHA 应相同。遇到其他提交先检查并合并，不能用 force 覆盖。
凭据由本机 Git/gh 已有登录提供，不将密码、token 或私钥写入仓库。
旧版首次推送的网络与空仓库说明已过时，不作为当前状态依据。

## 压缩包与课程提交

本地 `ProxUnroll-coursework-submission.zip` 是旧版 306 文件交付包，未包含本次扩展。
原包与 SHA256 文件保留，均不进入 Git。若最终选用新版，应另建新的文件名，
从最终提交生成后重新检查，不覆盖原包：

```bash
git archive --format=zip -9 --prefix=ProxUnroll-main/ \
  -o ProxUnroll-coursework-submission-NEW_VERSION.zip HEAD
shasum -a 256 ProxUnroll-coursework-submission-NEW_VERSION.zip \
  > ProxUnroll-coursework-submission-NEW_VERSION.zip.sha256
```

GitHub 同步不代表已提交课程平台。仍需确认组队/选题登记、最终报告版本、
姓名拼写、三人分工与实际理解，完成十分钟计时排练及两分钟问答准备。
具体课程要求核对见 `coursework/REQUIREMENTS_STATUS_20261009.md`。
