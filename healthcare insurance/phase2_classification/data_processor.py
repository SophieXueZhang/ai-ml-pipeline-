#!/usr/bin/env python3
"""
RVL-CDIP数据预处理器
筛选医保相关的5类文档：invoice, letter, email, memo, form
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from collections import Counter
import logging
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RVLCDIPProcessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
        # RVL-CDIP 16类映射
        self.original_classes = {
            0: 'letter',
            1: 'form', 
            2: 'email',
            3: 'handwritten',
            4: 'advertisement',
            5: 'scientific_report',
            6: 'scientific_publication',
            7: 'specification',
            8: 'file_folder',
            9: 'news_article',
            10: 'budget',
            11: 'invoice',
            12: 'presentation',
            13: 'questionnaire',
            14: 'resume',
            15: 'memo'
        }
        
        # 医保相关的5类
        self.target_classes = {
            'invoice': 0,     # 账单发票
            'letter': 1,      # 信件
            'email': 2,       # 邮件
            'memo': 3,        # 备忘录
            'form': 4         # 表单
        }
        
        # 原始类别到目标类别的映射
        self.class_mapping = {
            11: 0,  # invoice -> invoice
            0: 1,   # letter -> letter
            2: 2,   # email -> email
            15: 3,  # memo -> memo
            1: 4    # form -> form
        }
        
    def load_labels(self, split='train'):
        """加载标签文件"""
        label_file = self.data_dir / f"{split}.txt"
        if not label_file.exists():
            logger.error(f"标签文件不存在: {label_file}")
            return None
            
        # 读取标签文件
        data = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_path = parts[0]
                    label = int(parts[1])
                    data.append({'image_path': image_path, 'original_label': label})
        
        df = pd.DataFrame(data)
        logger.info(f"加载 {split} 数据: {len(df)} 条记录")
        return df
    
    def filter_target_classes(self, df):
        """筛选目标类别"""
        # 只保留目标类别
        target_original_labels = list(self.class_mapping.keys())
        filtered_df = df[df['original_label'].isin(target_original_labels)].copy()
        
        # 映射到新的标签
        filtered_df['label'] = filtered_df['original_label'].map(self.class_mapping)
        filtered_df['class_name'] = filtered_df['label'].map({v: k for k, v in self.target_classes.items()})
        
        logger.info(f"筛选后数据量: {len(filtered_df)}")
        
        # 统计各类别数量
        class_counts = filtered_df['class_name'].value_counts()
        logger.info("各类别分布:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count}")
            
        return filtered_df
    
    def create_balanced_subset(self, df, max_per_class=5000):
        """创建平衡的数据子集"""
        balanced_data = []
        
        for class_name in self.target_classes.keys():
            class_data = df[df['class_name'] == class_name]
            
            if len(class_data) > max_per_class:
                # 随机采样
                class_subset = class_data.sample(n=max_per_class, random_state=42)
                logger.info(f"{class_name}: 从 {len(class_data)} 采样到 {max_per_class}")
            else:
                class_subset = class_data
                logger.info(f"{class_name}: 保留全部 {len(class_data)} 条")
            
            balanced_data.append(class_subset)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序
        
        logger.info(f"平衡后总数据量: {len(balanced_df)}")
        return balanced_df
    
    def copy_images(self, df, split_name):
        """复制筛选后的图片到新目录"""
        target_dir = self.output_dir / "images" / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        fail_count = 0
        
        for idx, row in df.iterrows():
            src_path = self.data_dir / "images" / row['image_path']
            if src_path.exists():
                # 使用新的文件名：类别_原始文件名
                new_filename = f"{row['class_name']}_{Path(row['image_path']).name}"
                dst_path = target_dir / new_filename
                
                try:
                    shutil.copy2(src_path, dst_path)
                    # 更新DataFrame中的路径
                    df.at[idx, 'new_image_path'] = f"{split_name}/{new_filename}"
                    success_count += 1
                except Exception as e:
                    logger.error(f"复制文件失败 {src_path}: {e}")
                    fail_count += 1
            else:
                logger.warning(f"源文件不存在: {src_path}")
                fail_count += 1
        
        logger.info(f"图片复制完成 - 成功: {success_count}, 失败: {fail_count}")
        return df
    
    def save_processed_data(self, train_df, val_df, test_df):
        """保存处理后的数据"""
        # 保存CSV文件
        train_df.to_csv(self.output_dir / "train.csv", index=False)
        val_df.to_csv(self.output_dir / "val.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)
        
        # 保存类别信息
        class_info = {
            'target_classes': self.target_classes,
            'class_mapping': self.class_mapping,
            'original_classes': self.original_classes
        }
        
        import json
        with open(self.output_dir / "class_info.json", 'w', encoding='utf-8') as f:
            json.dump(class_info, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats = {
            'total_train': len(train_df),
            'total_val': len(val_df),
            'total_test': len(test_df),
            'train_distribution': train_df['class_name'].value_counts().to_dict(),
            'val_distribution': val_df['class_name'].value_counts().to_dict(),
            'test_distribution': test_df['class_name'].value_counts().to_dict()
        }
        
        with open(self.output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("数据保存完成")
        logger.info(f"训练集: {len(train_df)}")
        logger.info(f"验证集: {len(val_df)}")
        logger.info(f"测试集: {len(test_df)}")
    
    def process(self, max_per_class=5000):
        """主处理流程"""
        logger.info("开始处理RVL-CDIP数据集...")
        
        # 加载原始数据
        train_df = self.load_labels('train')
        val_df = self.load_labels('val')
        test_df = self.load_labels('test')
        
        if any(df is None for df in [train_df, val_df, test_df]):
            logger.error("加载数据失败")
            return False
        
        # 筛选目标类别
        train_filtered = self.filter_target_classes(train_df)
        val_filtered = self.filter_target_classes(val_df)
        test_filtered = self.filter_target_classes(test_df)
        
        # 创建平衡子集
        train_balanced = self.create_balanced_subset(train_filtered, max_per_class)
        val_balanced = self.create_balanced_subset(val_filtered, max_per_class//5)
        test_balanced = self.create_balanced_subset(test_filtered, max_per_class//5)
        
        # 复制图片（如果存在）
        if (self.data_dir / "images").exists():
            logger.info("复制训练图片...")
            train_balanced = self.copy_images(train_balanced, "train")
            logger.info("复制验证图片...")
            val_balanced = self.copy_images(val_balanced, "val")
            logger.info("复制测试图片...")
            test_balanced = self.copy_images(test_balanced, "test")
        else:
            logger.warning("图片目录不存在，跳过图片复制")
        
        # 保存处理后的数据
        self.save_processed_data(train_balanced, val_balanced, test_balanced)
        
        logger.info("🎉 数据处理完成！")
        return True

def main():
    """主函数"""
    # 获取数据目录
    base_dir = Path(__file__).parent.parent
    rvl_cdip_dir = base_dir / "data" / "rvl_cdip"
    
    if not rvl_cdip_dir.exists():
        logger.error(f"RVL-CDIP数据目录不存在: {rvl_cdip_dir}")
        logger.error("请先运行 python scripts/download_datasets.py")
        return False
    
    # 创建处理器并处理数据
    processor = RVLCDIPProcessor(rvl_cdip_dir)
    success = processor.process(max_per_class=5000)
    
    if success:
        logger.info("数据预处理成功完成！")
        logger.info(f"处理后的数据保存在: {processor.output_dir}")
    else:
        logger.error("数据预处理失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 