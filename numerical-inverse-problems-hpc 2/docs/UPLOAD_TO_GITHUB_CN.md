# 将项目上传到 GitHub：中文步骤

## 方法 A：只用网页（最简单）

1. 解压 `numerical-inverse-problems-hpc.zip`，**不要直接把 zip 当成源码上传**。
2. 登录 https://github.com ，进入 https://github.com/new 创建新仓库。
3. Repository name 填 `numerical-inverse-problems-hpc`。
4. 初次建议设置为 **Private**，确认代码版权和个人贡献后再改为 Public。创建时不要勾选额外生成 README/.gitignore，因为压缩包里已经有。
5. 点击 `Create repository`。
6. 在仓库首页点击 `uploading an existing file`；若不是空仓库，点 `Add file` → `Upload files`。
7. 把**解压后项目目录内**的 README.md、CONTRIBUTIONS.md、requirements.txt、.gitignore 及各子文件夹拖进去；注意 GitHub 网页拖拽上传会保留目录结构。
8. 填写提交信息 `Initial research code portfolio`，点击页面上的提交按钮。
9. 打开主页，确认 README 自动渲染，figures 图能打开、notebooks 可预览。

## 方法 B：用 Mac 终端（保留目录结构，最可靠）

在解压后的仓库目录执行：

```bash
cd /path/to/numerical-inverse-problems-hpc
git init
git add .
git commit -m "Initial research code portfolio"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/numerical-inverse-problems-hpc.git
git push -u origin main
```

**将 `YOUR_USERNAME` 改成你的 GitHub 用户名。** `git push` 需要先在 GitHub 上创建同名**空仓库**，并使用 GitHub 支持的认证方式登录。若提交失败，请先检查 Mac 上是否装好 Git、是否设置 Git user.name / user.email。

## 公开前检查

- 确认代码属于你可以公开发布的范围；如果属于科研组合作成果，先征得导师许可。
- 打开 `CONTRIBUTIONS.md`，逐条确认哪些是自己完成，哪些来自导师/参考代码。
- 检查电子邮箱、路径、账号、秘钥、隐私数据。
- 确保 README 没有夸大性能，未验证的脚本如实标为实验性。
- 你的 `main(1).pdf` 没被放进仓库，待确认作者和公开权限后再处理。

## 给教授

正式公开后，将仓库链接和单独撰写的 2–5 页英文个人研究总结 PDF 发给教授；README 和 CONTRIBUTIONS.md 可以作为项目索引与具体贡献介绍。

## 新增的个人研究汇报

`presentations/2026-05-13_cg_hpc_seminar_ru.pdf` 是你在 2026 年 5 月 13 日的俄文研究汇报。文件含个人电子邮箱，请在公开之前确认是否愿意展示，并确认研究组关于公开材料的要求。`docs/RESEARCH_SUMMARY.md` 是供教授阅读的英文总结，明确说明你的主要个人工作是**将数学算法编写成 Python 程序，以及实现 MPI 并行版本**。
