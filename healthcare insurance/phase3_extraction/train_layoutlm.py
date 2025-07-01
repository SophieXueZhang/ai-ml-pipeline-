#!/usr/bin/env python3
"""
LayoutLMv3训练脚本
用于医保表单关键信息抽取任务
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
    AdamW, get_scheduler, AutoConfig
)
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from seqeval.metrics import accuracy_score, f1_score as seq_f1_score, classification_report as seq_classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LayoutLMDataset(Dataset):
    """LayoutLMv3数据集"""
    
    def __init__(self, data_file, images_dir, processor, max_length=512):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 加载图片
        image_path = self.images_dir / sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法加载图片 {image_path}: {e}")
            # 创建空白图片作为替代
            image = Image.new('RGB', (224, 224), color='white')
        
        # 获取单词、边界框和标签
        words = sample['words']
        boxes = sample['boxes']
        labels = sample['labels']
        
        # 使用processor处理
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除batch维度
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        return encoding

class MedicalFieldExtractor:
    """医疗字段抽取器"""
    
    def __init__(self, label_map, medical_fields):
        self.label_map = label_map
        self.id2label = {v: k for k, v in label_map.items()}
        self.medical_fields = medical_fields
        
    def extract_entities(self, words, boxes, predictions):
        """从预测结果中抽取实体"""
        entities = {}
        current_entity = None
        current_words = []
        current_label = None
        
        for i, (word, box, pred_id) in enumerate(zip(words, boxes, predictions)):
            pred_label = self.id2label.get(pred_id, 'O')
            
            if pred_label.startswith('B-'):
                # 开始新实体
                if current_entity and current_words:
                    entities[current_entity] = {
                        'text': ' '.join(current_words),
                        'words': current_words.copy(),
                        'boxes': [boxes[i-len(current_words):i]],
                        'label': current_label
                    }
                
                current_entity = pred_label[2:]
                current_label = pred_label
                current_words = [word]
                
            elif pred_label.startswith('I-') and current_entity == pred_label[2:]:
                # 继续当前实体
                current_words.append(word)
                
            else:
                # 结束当前实体
                if current_entity and current_words:
                    entities[current_entity] = {
                        'text': ' '.join(current_words),
                        'words': current_words.copy(),
                        'boxes': [boxes[i-len(current_words):i]],
                        'label': current_label
                    }
                
                current_entity = None
                current_words = []
                current_label = None
        
        # 处理最后一个实体
        if current_entity and current_words:
            entities[current_entity] = {
                'text': ' '.join(current_words),
                'words': current_words.copy(),
                'boxes': [boxes[-len(current_words):]],
                'label': current_label
            }
        
        return entities
    
    def map_to_medical_fields(self, entities):
        """将实体映射到医疗字段"""
        medical_info = {}
        
        for entity_type, entity_data in entities.items():
            text = entity_data['text'].lower()
            
            # 检查是否匹配医疗字段
            for field_name, keywords in self.medical_fields.items():
                for keyword in keywords:
                    if keyword in text:
                        medical_info[field_name] = entity_data['text']
                        break
        
        return medical_info

class LayoutLMTrainer:
    """LayoutLMv3训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载标签信息
        with open(config['label_info_path'], 'r', encoding='utf-8') as f:
            self.label_info = json.load(f)
        
        self.label_map = self.label_info['label_map']
        self.id2label = self.label_info['id2label']
        self.num_labels = self.label_info['num_labels']
        
        # 初始化processor
        self.processor = LayoutLMv3Processor.from_pretrained(config['model_name'])
        
        # 医疗字段抽取器
        self.field_extractor = MedicalFieldExtractor(
            self.label_map, 
            self.label_info['medical_fields']
        )
        
    def create_data_loaders(self):
        """创建数据加载器"""
        logger.info("创建数据加载器...")
        
        # 训练集
        train_dataset = LayoutLMDataset(
            data_file=self.config['train_data'],
            images_dir=self.config['images_dir'],
            processor=self.processor,
            max_length=self.config['max_length']
        )
        
        # 验证集
        val_dataset = LayoutLMDataset(
            data_file=self.config['val_data'],
            images_dir=self.config['images_dir'],
            processor=self.processor,
            max_length=self.config['max_length']
        )
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def create_model(self):
        """创建模型"""
        logger.info("创建LayoutLMv3模型...")
        
        # 配置
        config = AutoConfig.from_pretrained(self.config['model_name'])
        config.num_labels = self.num_labels
        config.id2label = {int(k): v for k, v in self.id2label.items()}
        config.label2id = {v: int(k) for k, v in self.id2label.items()}
        
        # 模型
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.config['model_name'],
            config=config
        )
        model.to(self.device)
        
        return model
    
    def create_optimizer_and_scheduler(self, model, num_training_steps):
        """创建优化器和学习率调度器"""
        # 优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, scheduler):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="训练")
        
        for batch in progress_bar:
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, model, val_loader):
        """评估模型"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # 获取预测结果
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # 只考虑有效token (attention_mask == 1)
                active_mask = batch['attention_mask'].cpu().numpy() == 1
                
                for i in range(predictions.shape[0]):
                    active_preds = predictions[i][active_mask[i]].cpu().numpy()
                    active_labels = batch['labels'][i][active_mask[i]].cpu().numpy()
                    
                    # 过滤掉padding token (label == -100)
                    valid_indices = active_labels != -100
                    if valid_indices.any():
                        all_predictions.append(active_preds[valid_indices])
                        all_labels.append(active_labels[valid_indices])
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算F1分数
        flat_predictions = np.concatenate(all_predictions)
        flat_labels = np.concatenate(all_labels)
        
        f1 = f1_score(flat_labels, flat_predictions, average='weighted')
        
        return avg_loss, f1, all_predictions, all_labels
    
    def save_model(self, model, epoch, f1_score):
        """保存模型"""
        model_path = self.output_dir / f"layoutlmv3_epoch_{epoch}_f1_{f1_score:.4f}"
        
        # 保存模型和处理器
        model.save_pretrained(model_path)
        self.processor.save_pretrained(model_path)
        
        # 保存配置
        config_to_save = {
            'num_labels': self.num_labels,
            'label_info': self.label_info,
            'model_name': self.config['model_name'],
            'epoch': epoch,
            'f1_score': f1_score
        }
        
        with open(model_path / "training_config.json", 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到: {model_path}")
        return model_path
    
    def plot_label_distribution(self, all_labels, epoch):
        """绘制标签分布图"""
        # 统计标签分布
        flat_labels = np.concatenate(all_labels)
        unique_labels, counts = np.unique(flat_labels, return_counts=True)
        
        # 转换为标签名称
        label_names = [self.id2label.get(int(label), f'unknown_{label}') for label in unique_labels]
        
        plt.figure(figsize=(12, 6))
        plt.bar(label_names, counts)
        plt.title(f'Label Distribution - Epoch {epoch}')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f'label_distribution_epoch_{epoch}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def test_medical_field_extraction(self, model, test_samples=5):
        """测试医疗字段抽取功能"""
        logger.info("测试医疗字段抽取...")
        
        model.eval()
        
        # 创建测试样本
        test_words = ["Provider:", "Dr.", "John", "Smith", "Phone:", "555-123-4567", "Patient:", "Jane", "Doe"]
        test_boxes = [[50, 50, 120, 80], [130, 50, 160, 80], [170, 50, 220, 80], [230, 50, 290, 80],
                     [50, 100, 110, 130], [120, 100, 250, 130], [50, 150, 120, 180], [130, 150, 180, 180], [190, 150, 240, 180]]
        
        # 创建空白图片
        image = Image.new('RGB', (300, 200), color='white')
        
        # 处理
        encoding = self.processor(
            image,
            test_words,
            boxes=test_boxes,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
        
        # 移动到设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # 获取有效预测
            active_mask = encoding['attention_mask'][0].cpu().numpy() == 1
            active_predictions = predictions[0][active_mask].cpu().numpy()
            
            # 抽取实体
            entities = self.field_extractor.extract_entities(
                test_words, test_boxes, active_predictions[:len(test_words)]
            )
            
            # 映射到医疗字段
            medical_info = self.field_extractor.map_to_medical_fields(entities)
            
            logger.info("测试结果:")
            logger.info(f"实体: {entities}")
            logger.info(f"医疗信息: {medical_info}")
    
    def train(self):
        """主训练流程"""
        logger.info("开始训练LayoutLMv3...")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders()
        
        # 创建模型
        model = self.create_model()
        
        # 计算训练步数
        num_training_steps = len(train_loader) * self.config['epochs']
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, num_training_steps)
        
        # 训练循环
        best_f1 = 0
        best_model_path = None
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler)
            
            # 验证
            val_loss, val_f1, val_predictions, val_labels = self.evaluate(model, val_loader)
            
            logger.info(f"训练损失: {train_loss:.4f}")
            logger.info(f"验证损失: {val_loss:.4f}, 验证F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_path = self.save_model(model, epoch + 1, val_f1)
            
            # 绘制标签分布
            self.plot_label_distribution(val_labels, epoch + 1)
            
            # 测试医疗字段抽取
            self.test_medical_field_extraction(model)
        
        logger.info(f"🎉 训练完成！最佳F1分数: {best_f1:.4f}")
        logger.info(f"最佳模型保存在: {best_model_path}")
        
        return best_model_path, best_f1

def main():
    """主函数"""
    # 配置
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "funsd" / "processed"
    
    config = {
        'model_name': 'microsoft/layoutlmv3-base',
        'train_data': str(data_dir / "train.json"),
        'val_data': str(data_dir / "validation.json"),
        'images_dir': str(base_dir / "data" / "funsd"),
        'label_info_path': str(data_dir / "label_info.json"),
        'output_dir': str(base_dir / "models" / "extraction"),
        'batch_size': 8,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'epochs': 5,
        'warmup_steps': 1000,
        'max_length': 512
    }
    
    # 检查文件是否存在
    required_files = [
        config['train_data'],
        config['val_data'],
        config['label_info_path']
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"必需文件不存在: {file_path}")
            logger.error("请先运行FUNSD数据预处理脚本")
            return False
    
    # 创建训练器并训练
    trainer = LayoutLMTrainer(config)
    best_model_path, best_f1 = trainer.train()
    
    logger.info("训练成功完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 