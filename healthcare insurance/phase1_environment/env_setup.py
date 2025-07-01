#!/usr/bin/env python3
"""
环境设置脚本 - 医疗保险文档处理系统
检查GPU、下载预训练模型、验证环境配置
"""

import os
import sys
import torch
import platform
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_info():
    """检查系统信息"""
    logger.info("=== 系统信息检查 ===")
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA可用: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA不可用，将使用CPU运行（速度较慢）")
    
    # 检查MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        logger.info("Apple MPS可用")
    
    return torch.cuda.is_available() or torch.backends.mps.is_available()

def create_directories():
    """创建项目目录结构"""
    logger.info("=== 创建目录结构 ===")
    
    base_dir = Path(__file__).parent.parent
    directories = [
        "data/raw",
        "data/processed", 
        "data/rvl_cdip",
        "data/funsd",
        "data/cms_forms",
        "models/classification",
        "models/extraction", 
        "outputs/results",
        "outputs/visualizations",
        "phase1_environment",
        "phase2_classification",
        "phase3_extraction",
        "phase4_demo",
        "scripts"
    ]
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {full_path}")

def download_base_models():
    """下载基础预训练模型"""
    logger.info("=== 下载基础模型 ===")
    
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    # DiT模型 (文档分类)
    logger.info("下载DiT模型...")
    try:
        dit_model_name = "microsoft/dit-base-finetuned-rvlcdip"
        dit_model = AutoModel.from_pretrained(dit_model_name)
        dit_processor = AutoImageProcessor.from_pretrained(dit_model_name)
        
        # 保存到本地
        dit_local_path = models_dir / "dit_base"
        dit_model.save_pretrained(dit_local_path)
        dit_processor.save_pretrained(dit_local_path)
        logger.info(f"DiT模型已保存到: {dit_local_path}")
        
    except Exception as e:
        logger.error(f"下载DiT模型失败: {e}")
    
    # LayoutLMv3模型 (信息抽取)
    logger.info("下载LayoutLMv3模型...")
    try:
        layoutlm_model_name = "microsoft/layoutlmv3-base"
        layoutlm_model = AutoModel.from_pretrained(layoutlm_model_name)
        layoutlm_tokenizer = AutoTokenizer.from_pretrained(layoutlm_model_name)
        
        # 保存到本地
        layoutlm_local_path = models_dir / "layoutlmv3_base"
        layoutlm_model.save_pretrained(layoutlm_local_path)
        layoutlm_tokenizer.save_pretrained(layoutlm_local_path)
        logger.info(f"LayoutLMv3模型已保存到: {layoutlm_local_path}")
        
    except Exception as e:
        logger.error(f"下载LayoutLMv3模型失败: {e}")

def create_config_file():
    """创建配置文件"""
    logger.info("=== 创建配置文件 ===")
    
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config.py"
    
    config_content = '''"""
项目配置文件
"""
import torch
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# 设备配置
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 模型配置
MODELS_CONFIG = {
    "dit": {
        "model_name": "microsoft/dit-base-finetuned-rvlcdip",
        "local_path": MODELS_DIR / "dit_base",
        "num_classes": 5,  # 精简为5类
        "image_size": 224
    },
    "layoutlmv3": {
        "model_name": "microsoft/layoutlmv3-base", 
        "local_path": MODELS_DIR / "layoutlmv3_base",
        "max_length": 512
    }
}

# 数据集配置
DATASETS_CONFIG = {
    "rvl_cdip": {
        "url": "https://www.cs.cmu.edu/~aharley/rvl-cdip/",
        "local_path": DATA_DIR / "rvl_cdip",
        "selected_classes": ["invoice", "letter", "memo", "email", "form"]
    },
    "funsd": {
        "url": "https://guillaumejaume.github.io/FUNSD/",
        "local_path": DATA_DIR / "funsd"
    }
}

# 训练配置
TRAINING_CONFIG = {
    "classification": {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 3,
        "warmup_steps": 500
    },
    "extraction": {
        "batch_size": 8,
        "learning_rate": 1e-5,
        "epochs": 5,
        "warmup_steps": 1000
    }
}

# API配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".pdf", ".jpg", ".jpeg", ".png", ".tiff"]
}

# 目标字段配置 (CMS-1500表单)
TARGET_FIELDS = {
    "provider_name": "医疗提供者姓名",
    "provider_phone": "医疗提供者电话", 
    "provider_npi": "NPI号码",
    "patient_id": "患者ID",
    "patient_name": "患者姓名",
    "charge_total": "总费用",
    "diagnosis_code": "诊断代码",
    "service_date": "服务日期"
}
'''
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info(f"配置文件已创建: {config_path}")

def verify_installation():
    """验证安装是否成功"""
    logger.info("=== 验证安装 ===")
    
    try:
        # 测试transformers
        from transformers import pipeline
        logger.info("✓ Transformers正常")
        
        # 测试torch
        x = torch.randn(2, 3)
        logger.info("✓ PyTorch正常")
        
        # 测试PIL
        from PIL import Image
        logger.info("✓ PIL正常")
        
        # 测试datasets
        from datasets import Dataset
        logger.info("✓ Datasets正常")
        
        logger.info("所有核心依赖验证通过！")
        return True
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始环境设置...")
    
    # 检查系统
    has_gpu = check_system_info()
    
    # 创建目录
    create_directories()
    
    # 创建配置
    create_config_file()
    
    # 下载模型
    download_base_models()
    
    # 验证安装
    if verify_installation():
        logger.info("🎉 环境设置完成！")
        if has_gpu:
            logger.info("系统已准备好进行GPU加速训练")
        else:
            logger.info("系统将使用CPU进行训练（建议使用Colab/Kaggle获取免费GPU）")
    else:
        logger.error("环境设置失败，请检查错误信息")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 