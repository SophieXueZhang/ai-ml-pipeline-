#!/usr/bin/env python3
"""
数据集下载脚本
下载RVL-CDIP和FUNSD数据集
"""

import os
import sys
import requests
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, dest_path, desc="下载文件"):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def download_rvl_cdip():
    """下载RVL-CDIP数据集"""
    logger.info("=== 下载RVL-CDIP数据集 ===")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "rvl_cdip"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # RVL-CDIP数据集URL（由于文件很大，我们使用kaggle版本）
    kaggle_urls = {
        "images": "https://www.kaggle.com/datasets/sartajbhuvaji/rvl-cdip-images",
        "labels": "https://www.cs.cmu.edu/~aharley/rvl-cdip/labels.tar.gz"
    }
    
    # 下载标签文件
    labels_url = "https://www.cs.cmu.edu/~aharley/rvl-cdip/labels.tar.gz"
    labels_path = data_dir / "labels.tar.gz"
    
    if not labels_path.exists():
        logger.info("下载标签文件...")
        try:
            download_file(labels_url, labels_path, "下载RVL-CDIP标签")
            
            # 解压标签文件
            with tarfile.open(labels_path, 'r:gz') as tar:
                tar.extractall(data_dir)
            logger.info("标签文件解压完成")
            
        except Exception as e:
            logger.error(f"下载标签文件失败: {e}")
    
    # 创建说明文件
    readme_content = """
# RVL-CDIP数据集

由于RVL-CDIP数据集非常大（~400万张图片，~100GB），建议使用以下方式获取：

## 方式1：Kaggle下载（推荐）
1. 安装kaggle: pip install kaggle
2. 配置kaggle API token
3. 下载数据集: kaggle datasets download -d sartajbhuvaji/rvl-cdip-images
4. 解压到当前目录

## 方式2：直接下载
访问: https://www.cs.cmu.edu/~aharley/rvl-cdip/
下载完整数据集

## 数据集结构
```
rvl_cdip/
├── images/           # 图片文件
├── labels/           # 标签文件
├── train.txt         # 训练集列表
├── val.txt           # 验证集列表
└── test.txt          # 测试集列表
```

## 16个类别
0: letter, 1: form, 2: email, 3: handwritten, 4: advertisement, 
5: scientific report, 6: scientific publication, 7: specification, 
8: file folder, 9: news article, 10: budget, 11: invoice, 
12: presentation, 13: questionnaire, 14: resume, 15: memo

## 医保相关5类筛选
我们将重点关注以下5类：
- invoice (11)
- letter (0) 
- email (2)
- memo (15)
- form (1)
"""
    
    with open(data_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"RVL-CDIP数据集准备完成，请查看: {data_dir / 'README.md'}")

def download_funsd():
    """下载FUNSD数据集"""
    logger.info("=== 下载FUNSD数据集 ===")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "funsd"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # FUNSD数据集URL
    funsd_url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    zip_path = data_dir / "dataset.zip"
    
    if not zip_path.exists():
        logger.info("下载FUNSD数据集...")
        try:
            download_file(funsd_url, zip_path, "下载FUNSD数据集")
            
            # 解压文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info("FUNSD数据集解压完成")
            
        except Exception as e:
            logger.error(f"下载FUNSD数据集失败: {e}")
            # 创建替代说明
            readme_content = """
# FUNSD数据集下载失败

请手动下载FUNSD数据集：
1. 访问: https://guillaumejaume.github.io/FUNSD/
2. 下载dataset.zip
3. 解压到当前目录

## 数据集结构
```
funsd/
├── training_data/
│   ├── annotations/  # JSON标注文件
│   └── images/       # 图片文件
└── testing_data/
    ├── annotations/
    └── images/
```
"""
            with open(data_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
    
    logger.info(f"FUNSD数据集准备完成: {data_dir}")

def create_sample_data():
    """创建示例数据（用于测试）"""
    logger.info("=== 创建示例数据 ===")
    
    base_dir = Path(__file__).parent.parent
    sample_dir = base_dir / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例图片（简单的空白图片用于测试）
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # 创建示例文档图片
        samples = [
            ("invoice_sample.jpg", "INVOICE", "Sample invoice document"),
            ("letter_sample.jpg", "LETTER", "Sample letter document"),
            ("form_sample.jpg", "FORM", "Sample form document"),
            ("email_sample.jpg", "EMAIL", "Sample email document"),
            ("memo_sample.jpg", "MEMO", "Sample memo document")
        ]
        
        for filename, title, description in samples:
            # 创建800x600的白色图片
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # 绘制标题和文本
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("arial.ttf", 40)
                font_small = ImageFont.truetype("arial.ttf", 20)
            except:
                # 如果没有字体，使用默认字体
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # 绘制标题
            draw.text((50, 50), title, fill='black', font=font)
            draw.text((50, 150), description, fill='gray', font=font_small)
            
            # 绘制一些示例内容
            for i in range(5):
                draw.text((50, 250 + i*30), f"Sample text line {i+1}", fill='black', font=font_small)
            
            # 保存图片
            img.save(sample_dir / filename)
        
        logger.info(f"示例数据创建完成: {sample_dir}")
        
    except Exception as e:
        logger.error(f"创建示例数据失败: {e}")

def main():
    """主函数"""
    logger.info("开始下载数据集...")
    
    # 下载RVL-CDIP
    download_rvl_cdip()
    
    # 下载FUNSD
    download_funsd()
    
    # 创建示例数据
    create_sample_data()
    
    logger.info("🎉 数据集下载完成！")
    logger.info("请查看各数据集目录下的README.md文件获取详细信息")

if __name__ == "__main__":
    main() 