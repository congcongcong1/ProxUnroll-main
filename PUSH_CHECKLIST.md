# 推送与提交清单

本仓库已在本地初始化并完成首次提交。以下命令由本人自行在终端执行。

## 当前状态（已核对）

- 仓库根目录：`/Users/kaisar/Developer/COdec/ProxUnroll-main`
- 分支：`master`（尚无 upstream）
- 提交：`b76b7b6 Coursework: reproducible single-pixel imaging evaluation of ProxUnroll`，291 个文件
- **remote 已存在**：`origin` → `https://github.com/congcongcong1/ProxUnroll-main.git`
- 该 GitHub 仓库已确认状态：**私有、空仓库**（`isEmpty: true`，无默认分支），因此可以直接首次推送，无需 `pull --rebase`
- 本机 `gh` 已登录 github.com（账号 `congcongcong1`，HTTPS 协议）；git credential helper 为 `osxkeychain`

## 推送命令（已实测：必须先配代理 + 凭据）

实测结论：本机直连 `github.com:443` **不通**（`Failed to connect to github.com port 443
after 75002 ms`），但 macOS 系统代理可用（`127.0.0.1:65532`，HTTP/HTTPS/SOCKS 同一端口，
已由 `scutil --proxy` 确认，`nano` 进程在监听）。git 默认不读系统代理，因此首次推送
需要显式带上代理；同时钥匙串中没有 github.com 条目，需要先让 `gh` 给 git 装上凭据助手。

```bash
cd /Users/kaisar/Developer/COdec/ProxUnroll-main

# 1) 让 git 使用 gh 已登录的 token（写入全局 git 配置，只需执行一次）
gh auth setup-git

# 2) 首次推送：带系统代理（端口 65532）
git -c http.proxy=http://127.0.0.1:65532 \
    -c https.proxy=http://127.0.0.1:65532 \
    push -u origin master
```

如果之后还要反复推送，可把代理写进仓库配置，省去每次输入：

```bash
git config --local http.proxy http://127.0.0.1:65532
git config --local https.proxy http://127.0.0.1:65532
git push -u origin master

# 不想留下代理配置时删除
git config --local --unset http.proxy
git config --local --unset https.proxy
```

推送后核对：

```bash
git status --short --branch     # 应显示 ## master...origin/master
git ls-remote --heads origin    # 应列出 refs/heads/master
gh repo view --web              # 浏览器打开确认文件与提交
```

若想把主分支命名为 `main`，先改名再推送：

```bash
git branch -M main
git push -u origin main
```

### 其他备选通路

```bash
# 换 SSH（本机 ~/.ssh 下暂无公钥，需先生成并添加到 GitHub；代理配置见下）
git remote set-url origin git@github.com:congcongcong1/ProxUnroll-main.git
git push -u origin master

# 本机有 SOCKS 监听 1080，可让 SSH 走它（配合上面的 SSH 地址）
# ~/.ssh/config:
#   Host github.com
#     ProxyCommand nc -X 5 -x 127.0.0.1:1080 %h %p

# 若代理端口变了，用系统设置里的当前端口替换 65532：
scutil --proxy | grep -E 'HTTPPort|HTTPSPort'
```

### 关于权重文件

`weight/admm_proxunroll.pth` 与 `weight/hqs_proxunroll.pth` 各约 19 MB，已包含在提交中。
目标仓库是**私有**的，推送无合规顾虑；GitHub 单文件上限 100 MB，可正常上传。
若之后改为公开仓库，请先确认作者权重可再分发（原始仓库为公开仓库，LICENSE 已保留）。
如需排除权重，可执行：

```bash
git rm --cached weight/*.pth
printf 'weight/*.pth\n' >> .gitignore
git commit -m "Do not track pretrained weights; document download instead"
```

## 打包文件

- `ProxUnroll-coursework-submission.zip`（60 MB，306 个条目）
- 由 `git archive HEAD` 生成，内容与提交 `b76b7b6` 完全一致
- 已排除：`.venv/`、`tmp/`、`__pycache__/`、`.pytest_cache/`、`.chart-data-*/`、`.DS_Store`、`.git/`
- 解压后顶层目录为 `ProxUnroll-main/`；含中文文件名 `coursework/答辩准备.md`，macOS 用
  `ditto -x -k <zip> <目标目录>` 或访达双击解压（GNU `unzip` 可能显示乱码并报写入错误）
- SHA256 见下方「校验值」

解压后可直接查看交付物：

- `ProxUnroll-main/output/pdf/technical_report.pdf`（6 页）
- `ProxUnroll-main/output/pdf/literature_review.pdf`
- `ProxUnroll-main/output/presentation/course_presentation.pptx`
- `ProxUnroll-main/coursework/PRESENTATION_GUIDE.md`、`coursework/答辩准备.md`
- `ProxUnroll-main/COURSEWORK.md`（复现命令入口）

## 校验值

zip 由 `git archive HEAD` 生成，其 SHA256 记录在同目录的
`ProxUnroll-coursework-submission.zip.sha256`（zip 本身不纳入版本控制，
避免"内容随自身哈希变化"的循环）。校验方式：

```bash
cd /Users/kaisar/Developer/COdec/ProxUnroll-main
shasum -a 256 -c ProxUnroll-coursework-submission.zip.sha256
```

需要重新生成时：

```bash
git archive --format=zip -9 --prefix=ProxUnroll-main/ \
  -o ProxUnroll-coursework-submission.zip HEAD
shasum -a 256 ProxUnroll-coursework-submission.zip \
  > ProxUnroll-coursework-submission.zip.sha256
```

## 提交前仍建议确认

1. 英文姓名拼写与老师指定的封面/模板要求（`COURSEWORK.md` 已标注模板未提供）。
2. 三位成员各自能解释测量方程、伴随算子、DCT 先验、HQS/ADMM 循环。
3. 老师若要求仓库公开，记得在 GitHub 设置里把 Private 改为 Public 并确认权重可再分发。
