#!/usr/bin/env python3
"""
示例模型定义
展示如何在项目中定义和使用深度学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

class SimpleClassifier(nn.Module):
    """简单的全连接分类器"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class CNNClassifier(nn.Module):
    """卷积神经网络分类器"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(CNNClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNetClassifier(nn.Module):
    """基于ResNet的分类器"""
    
    def __init__(self, num_classes=10, pretrained=True, architecture='resnet50'):
        super(ResNetClassifier, self).__init__()
        
        if architecture == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
        elif architecture == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的架构: {architecture}")
        
        # 替换最后的分类层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def create_model(config):
    """根据配置创建模型"""
    model_name = config.get('name', 'simple')
    num_classes = config.get('num_classes', 10)
    
    if model_name == 'simple':
        input_size = config.get('input_size', 784)
        hidden_size = config.get('hidden_size', 128)
        return SimpleClassifier(input_size, hidden_size, num_classes)
    
    elif model_name == 'cnn':
        input_channels = config.get('input_channels', 3)
        return CNNClassifier(num_classes, input_channels)
    
    elif model_name in ['resnet50', 'resnet18']:
        pretrained = config.get('pretrained', True)
        return ResNetClassifier(num_classes, pretrained, model_name)
    
    else:
        raise ValueError(f"未知的模型类型: {model_name}")

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size):
    """打印模型摘要"""
    print(f"模型: {model.__class__.__name__}")
    print(f"参数数量: {count_parameters(model):,}")
    
    # 计算模型大小
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
    print(f"模型大小: {total_size_mb:.2f} MB")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        try:
            output = model(dummy_input)
            print(f"输入形状: {dummy_input.shape}")
            print(f"输出形状: {output.shape}")
        except Exception as e:
            print(f"前向传播测试失败: {e}")

if __name__ == "__main__":
    # 测试不同的模型
    print("=" * 50)
    print("模型测试")
    print("=" * 50)
    
    # 简单分类器
    print("\n1. 简单分类器 (MNIST)")
    simple_model = SimpleClassifier(784, 128, 10)
    model_summary(simple_model, (784,))
    
    # CNN分类器
    print("\n2. CNN分类器 (CIFAR-10)")
    cnn_model = CNNClassifier(10, 3)
    model_summary(cnn_model, (3, 32, 32))
    
    # ResNet分类器
    print("\n3. ResNet-18分类器")
    resnet_model = ResNetClassifier(10, pretrained=False, architecture='resnet18')
    model_summary(resnet_model, (3, 224, 224)) 