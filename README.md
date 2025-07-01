# AI ML Pipeline - 本地开发 + Colab GPU

这个项目让您可以在本地编写和管理代码，同时利用Google Colab的免费GPU资源进行模型训练和推理。

## 快速开始

### 1. 本地环境设置

```bash
# 安装基础依赖
pip install -r requirements.txt

# 设置开发环境
python setup_dev_env.py
```

### 2. Colab连接方式

#### 方式一：文件同步 (推荐)
1. 运行 `colab_sync.ipynb` 在Colab中
2. 自动同步本地代码到Colab
3. 在Colab中使用GPU执行训练

#### 方式二：SSH隧道连接
1. 使用 `colab_ssh_setup.ipynb` 建立SSH连接
2. 直接从本地IDE连接到Colab环境

#### 方式三：Jupyter远程连接
1. 在Colab启动Jupyter服务器
2. 本地通过端口转发连接

## 项目结构

```
ai-ml-pipeline/
├── src/                 # 本地开发代码
├── notebooks/           # Colab笔记本
├── configs/            # 配置文件
├── data/               # 数据文件
├── models/             # 模型文件
├── colab_templates/    # Colab模板
└── utils/              # 工具函数
```

## 使用说明

### 本地开发
- 在 `src/` 目录下编写您的ML代码
- 使用 `configs/` 管理不同的训练配置
- 本地测试和调试小规模代码

### Colab执行
- 使用模板笔记本自动同步代码
- 利用Colab的T4/A100 GPU进行训练
- 结果自动保存回本地

## 特性
- 🚀 无缝的本地-Colab工作流
- 🔧 自动代码同步
- 📊 实时训练监控
- �� 自动模型保存
- 🎯 多GPU支持 