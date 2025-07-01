# 医疗保险文档智能处理系统

## 🎯 项目概述

这是一个端到端的医疗保险文档处理系统，能够：
- 自动分类医保文档类型（理赔申请、账单、邮件等）
- 智能抽取表单关键信息（患者信息、医生信息、费用等）
- 输出结构化JSON数据，直接对接Power BI或数据库
- 提供FastAPI RESTful服务接口

## 🚀 快速开始

### 1. 环境设置
```bash
pip install -r requirements.txt
python scripts/env_setup.py
```

### 2. 数据准备
```bash
python scripts/download_datasets.py
python scripts/prepare_data.py
```

### 3. 模型训练
```bash
# 文档分类模型 (DiT)
python phase2_classification/train_dit.py

# 信息抽取模型 (LayoutLMv3)
python phase3_extraction/train_layoutlm.py
```

### 4. 启动服务
```bash
python phase4_demo/app.py
```

## 📁 项目结构

```
healthcare-insurance/
├── phase1_environment/     # 环境配置与基础设置
├── phase2_classification/  # 文档分类 (DiT fine-tune)
├── phase3_extraction/      # 信息抽取 (LayoutLMv3 fine-tune)  
├── phase4_demo/           # 端到端演示与API服务
├── data/                  # 数据集存储目录
├── models/                # 训练好的模型
├── outputs/               # 输出结果
└── scripts/               # 通用脚本
```

## 📊 性能指标

- **文档分类准确率**: ≥94%
- **信息抽取F1分数**: ≥80%
- **API响应时间**: ≤3秒
- **支持文档格式**: PDF, JPEG, PNG, TIFF

## 🔧 技术栈

- **深度学习**: PyTorch, Transformers, LayoutLMv3, DiT
- **数据处理**: Pandas, NumPy, OpenCV
- **API服务**: FastAPI, Uvicorn
- **可视化**: Power BI, Matplotlib, Seaborn
- **部署**: Docker, Azure (可选) 