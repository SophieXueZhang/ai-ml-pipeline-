#!/usr/bin/env python3
"""
DiT (Document Image Transformer) 训练脚本
用于医保文档分类任务
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoImageProcessor, AutoConfig,
    AdamW, get_scheduler
)
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthcareDocumentDataset(Dataset):
    """医保文档数据集"""
    
    def __init__(self, csv_path, images_dir, processor, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载图片
        if 'new_image_path' in row and pd.notna(row['new_image_path']):
            image_path = self.images_dir.parent / "images" / row['new_image_path']
        else:
            image_path = self.images_dir / row['image_path']
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法加载图片 {image_path}: {e}")
            # 创建一个空白图片作为替代
            image = Image.new('RGB', (224, 224), color='white')
        
        # 预处理图片
        if self.processor:
            inputs = self.processor(image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # 移除batch维度
        else:
            # 如果没有processor，使用基本的转换
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        # 获取标签
        label = int(row['label'])
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DiTClassifier(nn.Module):
    """基于DiT的文档分类器"""
    
    def __init__(self, model_name_or_path, num_classes=5):
        super().__init__()
        
        # 加载DiT模型
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.dit = AutoModel.from_pretrained(model_name_or_path)
        
        # 分类头
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, pixel_values, labels=None):
        # DiT前向传播
        outputs = self.dit(pixel_values=pixel_values)
        
        # 使用[CLS] token或池化后的表示
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # 使用最后一层的平均池化
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier[-1].out_features), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
        }

class DiTTrainer:
    """DiT训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载处理器
        self.processor = AutoImageProcessor.from_pretrained(config['model_name'])
        
        # 加载类别信息
        with open(config['class_info_path'], 'r', encoding='utf-8') as f:
            self.class_info = json.load(f)
        self.num_classes = len(self.class_info['target_classes'])
        
    def create_data_loaders(self):
        """创建数据加载器"""
        logger.info("创建数据加载器...")
        
        # 训练集
        train_dataset = HealthcareDocumentDataset(
            csv_path=self.config['train_csv'],
            images_dir=self.config['images_dir'],
            processor=self.processor
        )
        
        # 验证集
        val_dataset = HealthcareDocumentDataset(
            csv_path=self.config['val_csv'], 
            images_dir=self.config['images_dir'],
            processor=self.processor
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
        logger.info("创建DiT分类模型...")
        
        model = DiTClassifier(
            model_name_or_path=self.config['model_name'],
            num_classes=self.num_classes
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
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="训练")
        
        for batch in progress_bar:
            # 移动到设备
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs['loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, model, val_loader):
        """评估模型"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def save_model(self, model, epoch, accuracy):
        """保存模型"""
        model_path = self.output_dir / f"dit_classifier_epoch_{epoch}_acc_{accuracy:.4f}"
        model_path.mkdir(exist_ok=True)
        
        # 保存模型
        model.dit.save_pretrained(model_path / "dit")
        self.processor.save_pretrained(model_path / "processor")
        
        # 保存分类头
        torch.save(model.classifier.state_dict(), model_path / "classifier.pth")
        
        # 保存配置
        config_to_save = {
            'num_classes': self.num_classes,
            'class_info': self.class_info,
            'model_name': self.config['model_name'],
            'epoch': epoch,
            'accuracy': accuracy
        }
        
        with open(model_path / "training_config.json", 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到: {model_path}")
        return model_path
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """绘制混淆矩阵"""
        class_names = list(self.class_info['target_classes'].keys())
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图片
        plt.savefig(self.output_dir / f'confusion_matrix_epoch_{epoch}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """主训练流程"""
        logger.info("开始训练DiT分类器...")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders()
        
        # 创建模型
        model = self.create_model()
        
        # 计算训练步数
        num_training_steps = len(train_loader) * self.config['epochs']
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, num_training_steps)
        
        # 训练循环
        best_accuracy = 0
        best_model_path = None
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, scheduler)
            
            # 验证
            val_loss, val_acc, val_predictions, val_labels = self.evaluate(model, val_loader)
            
            logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_path = self.save_model(model, epoch + 1, val_acc)
            
            # 绘制混淆矩阵
            self.plot_confusion_matrix(val_labels, val_predictions, epoch + 1)
            
            # 打印分类报告
            class_names = list(self.class_info['target_classes'].keys())
            report = classification_report(val_labels, val_predictions, 
                                         target_names=class_names, digits=4)
            logger.info(f"\n分类报告:\n{report}")
        
        logger.info(f"🎉 训练完成！最佳准确率: {best_accuracy:.4f}")
        logger.info(f"最佳模型保存在: {best_model_path}")
        
        return best_model_path, best_accuracy

def main():
    """主函数"""
    # 配置
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "rvl_cdip" / "processed"
    
    config = {
        'model_name': str(base_dir / "models" / "dit_base"),
        'train_csv': str(data_dir / "train.csv"),
        'val_csv': str(data_dir / "val.csv"),
        'images_dir': str(data_dir),
        'class_info_path': str(data_dir / "class_info.json"),
        'output_dir': str(base_dir / "models" / "classification"),
        'batch_size': 16,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'epochs': 3,
        'warmup_steps': 500
    }
    
    # 检查文件是否存在
    required_files = [
        config['train_csv'],
        config['val_csv'],
        config['class_info_path']
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"必需文件不存在: {file_path}")
            logger.error("请先运行数据预处理脚本")
            return False
    
    # 如果本地DiT模型不存在，使用在线版本
    if not Path(config['model_name']).exists():
        logger.warning("本地DiT模型不存在，使用在线版本")
        config['model_name'] = "microsoft/dit-base-finetuned-rvlcdip"
    
    # 创建训练器并训练
    trainer = DiTTrainer(config)
    best_model_path, best_accuracy = trainer.train()
    
    logger.info("训练成功完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 