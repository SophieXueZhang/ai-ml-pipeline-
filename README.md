# AI项目 - 本地开发 + Colab GPU 训练

这个项目支持本地开发和Colab GPU训练的无缝结合。

## 🚀 快速开始

### 1. 本地开发流程

```bash
# 编辑代码
vim your_code.py

# 提交更改
git add .
git commit -m "your changes"
git push
```

### 2. Colab GPU 训练

1. 打开 `colab_template.ipynb`
2. 上传到 Google Colab
3. 修改第2个代码块中的 `REPO_URL` 为你的GitHub仓库地址
4. 在 Runtime → Change runtime type 中选择 GPU
5. 运行所有代码块

## 📁 项目结构

```
ai/
├── healthcare insurance/          # 医疗保险ML项目
│   ├── phase1_environment/       # 环境设置
│   ├── phase2_classification/    # 文档分类
│   ├── phase3_extraction/        # 信息提取
│   ├── phase4_demo/             # 演示应用
│   └── requirements.txt         # 项目依赖
├── lora/                        # LoRA相关实验
├── colab_template.ipynb         # Colab模板
├── requirements.txt             # 根目录依赖
└── README.md                    # 本文档
```

## 🔧 设置GitHub仓库

1. 在GitHub创建新仓库
2. 添加远程仓库：
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

## 💡 使用技巧

- **私有仓库**: 使用Personal Access Token替换URL中的用户名密码
- **大文件**: 数据集和模型权重存储在Google Drive，不要提交到Git
- **开发循环**: 本地改代码 → `git push` → Colab运行「克隆/更新代码」→ GPU训练
- **会话管理**: Colab 12小时后断开，重要结果保存到Drive

## 🎯 主要功能

### Healthcare Insurance项目
- 文档分类（DiT模型）
- 信息提取（LayoutLM）
- Web演示界面
- 数据验证工具

### LoRA实验
- LLaVA模型微调
- 环境配置脚本

## 📚 更多资源

- [Colab GPU使用指南](https://colab.research.google.com/notebooks/gpu.ipynb)
- [Git基础教程](https://git-scm.com/docs/gittutorial)
- [GitHub Personal Access Token设置](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) 