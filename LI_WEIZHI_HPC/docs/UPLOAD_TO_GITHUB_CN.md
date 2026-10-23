# 如何发布到 GitHub（中文）

1. 下载并解压 `LI_WEIZHI_HPC_PhD_Portfolio_GitHub.zip`。
2. 打开 https://github.com/new，仓库名建议 `numerical-inverse-problems-hpc`。
3. 第一次上传建议设为 Private，确认研究组材料和汇报可公开后再改 Public。
4. 创建仓库时**不要勾选 Initialize with a README**，因为本项目已经有 `README.md`。
5. 新仓库页面点击 `uploading an existing file`；如果已有文件，点 `Add file` → `Upload files`。
6. 上传的是**解压后的各个文件与子文件夹**，不要只上传 ZIP 文件。提交（Commit changes）。
7. 检查 README 能正常显示三张图和报告链接，随后复制仓库 URL 发给教授。
8. 重要：请保留代码与报告中区分 *model-based prediction* 与 *measured speedup* 的说明。

如网页不支持拖入整个文件夹，可以在本地终端从项目根目录执行：

```bash
git init
git add .
git commit -m "Initial research portfolio"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/numerical-inverse-problems-hpc.git
git push -u origin main
```

你必须先在 GitHub 创建对应空仓库、配置 Git 用户和认证。`YOUR-USERNAME` 改成自己的账号。
