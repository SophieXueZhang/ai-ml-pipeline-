#!/usr/bin/env python3
"""
FUNSD数据集处理器
处理表单理解数据，准备LayoutLMv3训练
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from collections import defaultdict
import cv2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FUNSDProcessor:
    """FUNSD数据集处理器"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
        # FUNSD标签映射
        self.label_map = {
            'O': 0,          # Outside
            'B-HEADER': 1,   # Begin Header
            'I-HEADER': 2,   # Inside Header
            'B-QUESTION': 3, # Begin Question
            'I-QUESTION': 4, # Inside Question
            'B-ANSWER': 5,   # Begin Answer
            'I-ANSWER': 6,   # Inside Answer
            'B-OTHER': 7,    # Begin Other
            'I-OTHER': 8     # Inside Other
        }
        
        self.id2label = {v: k for k, v in self.label_map.items()}
        
        # 医疗表单相关字段映射（扩展FUNSD用于医疗场景）
        self.medical_fields = {
            'provider_name': ['doctor', 'physician', 'provider', 'clinic', 'hospital'],
            'provider_phone': ['phone', 'tel', 'telephone', 'contact'],
            'provider_npi': ['npi', 'provider id', 'physician id'],
            'patient_id': ['patient id', 'member id', 'subscriber id'],
            'patient_name': ['patient', 'name', 'subscriber', 'member'],
            'charge_total': ['total', 'amount', 'charge', 'cost', 'fee'],
            'diagnosis_code': ['diagnosis', 'dx', 'icd', 'code'],
            'service_date': ['date', 'service date', 'dos', 'treatment date']
        }
    
    def load_funsd_annotation(self, annotation_path):
        """加载FUNSD标注文件"""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_words_and_boxes(self, annotation_data, image_path):
        """从标注数据中提取单词和边界框"""
        words = []
        boxes = []
        labels = []
        
        # 获取图片尺寸
        try:
            image = Image.open(image_path)
            img_width, img_height = image.size
        except Exception as e:
            logger.warning(f"无法加载图片 {image_path}: {e}")
            img_width, img_height = 1000, 1000  # 默认尺寸
        
        for item in annotation_data['form']:
            # 获取实体标签
            entity_label = item.get('label', 'other').lower()
            
            # 映射到BIO标记
            if 'header' in entity_label:
                bio_prefix = 'HEADER'
            elif 'question' in entity_label:
                bio_prefix = 'QUESTION'  
            elif 'answer' in entity_label:
                bio_prefix = 'ANSWER'
            else:
                bio_prefix = 'OTHER'
            
            # 处理单词
            first_word = True
            for word_info in item.get('words', []):
                word_text = word_info.get('text', '').strip()
                if not word_text:
                    continue
                
                # BIO标记
                if first_word:
                    label = f'B-{bio_prefix}'
                    first_word = False
                else:
                    label = f'I-{bio_prefix}'
                
                # 边界框 (相对坐标转换)
                bbox = word_info.get('box', [0, 0, 100, 100])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # 确保坐标在合理范围内
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(x1, min(x2, img_width))
                    y2 = max(y1, min(y2, img_height))
                    
                    # 转换为LayoutLMv3格式 (normalized coordinates)
                    norm_box = [
                        int(1000 * x1 / img_width),
                        int(1000 * y1 / img_height),
                        int(1000 * x2 / img_width),
                        int(1000 * y2 / img_height)
                    ]
                else:
                    norm_box = [0, 0, 1000, 1000]
                
                words.append(word_text)
                boxes.append(norm_box)
                labels.append(self.label_map.get(label, 0))
        
        return words, boxes, labels
    
    def create_medical_synthetic_data(self, base_words, base_boxes, base_labels, num_samples=50):
        """创建医疗表单合成数据"""
        synthetic_data = []
        
        # 医疗表单模板
        medical_templates = [
            {
                'provider_name': 'Dr. John Smith',
                'provider_phone': '(555) 123-4567',
                'provider_npi': '1234567890',
                'patient_id': 'P123456',
                'patient_name': 'Jane Doe',
                'charge_total': '$150.00',
                'diagnosis_code': 'M79.3',
                'service_date': '01/15/2024'
            },
            {
                'provider_name': 'City Medical Center',
                'provider_phone': '555-987-6543',
                'provider_npi': '9876543210',
                'patient_id': 'MED789',
                'patient_name': 'Robert Johnson',
                'charge_total': '$275.50',
                'diagnosis_code': 'J06.9',
                'service_date': '02/20/2024'
            }
        ]
        
        import random
        
        for i in range(num_samples):
            template = random.choice(medical_templates)
            
            # 随机修改模板值
            sample = {}
            for field, value in template.items():
                if field == 'provider_name':
                    names = ['Dr. Smith', 'Dr. Johnson', 'Dr. Brown', 'Medical Center', 'Clinic']
                    sample[field] = random.choice(names)
                elif field == 'provider_phone':
                    sample[field] = f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}"
                elif field == 'provider_npi':
                    sample[field] = str(random.randint(1000000000, 9999999999))
                elif field == 'patient_id':
                    sample[field] = f"P{random.randint(10000, 99999)}"
                elif field == 'patient_name':
                    first_names = ['John', 'Jane', 'Robert', 'Mary', 'David', 'Sarah']
                    last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Moore']
                    sample[field] = f"{random.choice(first_names)} {random.choice(last_names)}"
                elif field == 'charge_total':
                    amount = random.randint(50, 500)
                    sample[field] = f"${amount}.00"
                elif field == 'diagnosis_code':
                    codes = ['M79.3', 'J06.9', 'K59.0', 'R50.9', 'Z00.00']
                    sample[field] = random.choice(codes)
                elif field == 'service_date':
                    month = random.randint(1, 12)
                    day = random.randint(1, 28)
                    sample[field] = f"{month:02d}/{day:02d}/2024"
                else:
                    sample[field] = value
            
            synthetic_data.append(sample)
        
        return synthetic_data
    
    def process_split(self, split_name):
        """处理单个数据集分割"""
        logger.info(f"处理 {split_name} 数据...")
        
        split_dir = self.data_dir / f"{split_name}_data"
        if not split_dir.exists():
            logger.warning(f"{split_name} 目录不存在: {split_dir}")
            return []
        
        annotations_dir = split_dir / "annotations"
        images_dir = split_dir / "images"
        
        processed_data = []
        
        # 处理每个标注文件
        if annotations_dir.exists():
            for annotation_file in annotations_dir.glob("*.json"):
                try:
                    # 加载标注
                    annotation_data = self.load_funsd_annotation(annotation_file)
                    
                    # 对应的图片文件
                    image_file = images_dir / f"{annotation_file.stem}.png"
                    if not image_file.exists():
                        logger.warning(f"图片文件不存在: {image_file}")
                        continue
                    
                    # 提取单词、边界框和标签
                    words, boxes, labels = self.extract_words_and_boxes(
                        annotation_data, image_file
                    )
                    
                    if words:
                        processed_data.append({
                            'id': annotation_file.stem,
                            'image_path': str(image_file.relative_to(self.data_dir)),
                            'words': words,
                            'boxes': boxes,
                            'labels': labels
                        })
                        
                except Exception as e:
                    logger.error(f"处理文件失败 {annotation_file}: {e}")
        
        logger.info(f"{split_name} 数据处理完成: {len(processed_data)} 个样本")
        return processed_data
    
    def create_cms_synthetic_data(self, num_samples=100):
        """创建CMS-1500表单合成数据"""
        logger.info("创建CMS-1500合成数据...")
        
        synthetic_samples = []
        
        # 创建简单的合成CMS-1500表单数据
        for i in range(num_samples):
            # 模拟CMS-1500表单的关键字段
            sample_data = {
                'id': f'cms_synthetic_{i:03d}',
                'image_path': f'synthetic/cms_{i:03d}.png',  # 虚拟路径
                'words': [
                    'HEALTH', 'INSURANCE', 'CLAIM', 'FORM',
                    'Provider:', 'Dr.', 'John', 'Smith',
                    'Phone:', '555-123-4567',
                    'NPI:', '1234567890',
                    'Patient:', 'Jane', 'Doe',
                    'ID:', 'P123456',
                    'Total:', '$250.00',
                    'Date:', '01/15/2024',
                    'Diagnosis:', 'M79.3'
                ],
                'boxes': [
                    [50, 50, 150, 80], [160, 50, 280, 80], [290, 50, 380, 80], [390, 50, 450, 80],  # 标题
                    [50, 150, 120, 180], [130, 150, 160, 180], [170, 150, 220, 180], [230, 150, 290, 180],  # Provider
                    [50, 200, 110, 230], [120, 200, 250, 230],  # Phone
                    [50, 250, 90, 280], [100, 250, 200, 280],  # NPI
                    [50, 300, 120, 330], [130, 300, 180, 330], [190, 300, 240, 330],  # Patient
                    [50, 350, 80, 380], [90, 350, 150, 380],  # ID
                    [50, 400, 100, 430], [110, 400, 180, 430],  # Total
                    [50, 450, 100, 480], [110, 450, 190, 480],  # Date
                    [50, 500, 130, 530], [140, 500, 200, 530]  # Diagnosis
                ],
                'labels': [
                    7, 7, 7, 7,  # OTHER (title)
                    3, 5, 5, 5,  # QUESTION, ANSWER, ANSWER, ANSWER (provider)
                    3, 5,        # QUESTION, ANSWER (phone)
                    3, 5,        # QUESTION, ANSWER (npi)
                    3, 5, 5,     # QUESTION, ANSWER, ANSWER (patient)
                    3, 5,        # QUESTION, ANSWER (id)
                    3, 5,        # QUESTION, ANSWER (total)
                    3, 5,        # QUESTION, ANSWER (date)
                    3, 5         # QUESTION, ANSWER (diagnosis)
                ]
            }
            
            synthetic_samples.append(sample_data)
        
        logger.info(f"创建了 {len(synthetic_samples)} 个CMS-1500合成样本")
        return synthetic_samples
    
    def save_processed_data(self, train_data, val_data, test_data):
        """保存处理后的数据"""
        logger.info("保存处理后的数据...")
        
        # 保存为JSON格式（LayoutLMv3训练格式）
        datasets = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, data in datasets.items():
            output_file = self.output_dir / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"保存 {split_name}: {len(data)} 个样本 -> {output_file}")
        
        # 保存标签映射
        label_info = {
            'label_map': self.label_map,
            'id2label': self.id2label,
            'num_labels': len(self.label_map),
            'medical_fields': self.medical_fields
        }
        
        with open(self.output_dir / "label_info.json", 'w', encoding='utf-8') as f:
            json.dump(label_info, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats = {
            'total_train': len(train_data),
            'total_val': len(val_data),
            'total_test': len(test_data),
            'label_distribution': self._calculate_label_distribution(train_data + val_data + test_data)
        }
        
        with open(self.output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("数据保存完成")
    
    def _calculate_label_distribution(self, all_data):
        """计算标签分布"""
        label_counts = defaultdict(int)
        
        for sample in all_data:
            for label in sample['labels']:
                label_name = self.id2label.get(label, 'unknown')
                label_counts[label_name] += 1
        
        return dict(label_counts)
    
    def process(self):
        """主处理流程"""
        logger.info("开始处理FUNSD数据集...")
        
        # 处理原始FUNSD数据
        train_data = self.process_split('training')
        test_data = self.process_split('testing')
        
        # 从训练数据中划分验证集
        import random
        random.seed(42)
        
        if train_data:
            # 80-20分割
            split_idx = int(0.8 * len(train_data))
            random.shuffle(train_data)
            
            funsd_train = train_data[:split_idx]
            funsd_val = train_data[split_idx:]
        else:
            funsd_train = []
            funsd_val = []
        
        # 创建医疗表单合成数据
        cms_synthetic_data = self.create_cms_synthetic_data(num_samples=100)
        
        # 合并数据
        final_train = funsd_train + cms_synthetic_data[:80]  # 大部分合成数据用于训练
        final_val = funsd_val + cms_synthetic_data[80:]      # 少部分用于验证
        final_test = test_data
        
        # 如果没有测试数据，从验证中分一些
        if not final_test and final_val:
            split_idx = len(final_val) // 2
            final_test = final_val[split_idx:]
            final_val = final_val[:split_idx]
        
        # 保存处理后的数据
        self.save_processed_data(final_train, final_val, final_test)
        
        logger.info("🎉 FUNSD数据处理完成！")
        logger.info(f"训练集: {len(final_train)}")
        logger.info(f"验证集: {len(final_val)}")
        logger.info(f"测试集: {len(final_test)}")
        
        return True

def main():
    """主函数"""
    # 获取数据目录
    base_dir = Path(__file__).parent.parent
    funsd_dir = base_dir / "data" / "funsd"
    
    if not funsd_dir.exists():
        logger.error(f"FUNSD数据目录不存在: {funsd_dir}")
        logger.error("请先运行 python scripts/download_datasets.py")
        return False
    
    # 创建处理器并处理数据
    processor = FUNSDProcessor(funsd_dir)
    success = processor.process()
    
    if success:
        logger.info("FUNSD数据预处理成功完成！")
        logger.info(f"处理后的数据保存在: {processor.output_dir}")
    else:
        logger.error("FUNSD数据预处理失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 