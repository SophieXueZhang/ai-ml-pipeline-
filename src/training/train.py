#!/usr/bin/env python3
"""
训练脚本模板
用于在Colab GPU环境中训练深度学习模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_config(config_path='configs/train_config.json'):
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ 使用CPU训练")
    return device

def create_model(config):
    """创建模型"""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    
    if model_name == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(pretrained=config['model']['pretrained'])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # 添加其他模型
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model

def create_optimizer(model, config):
    """创建优化器"""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='训练中')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc='验证中'):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def save_checkpoint(model, optimizer, epoch, loss, path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✅ 检查点已保存: {path}")

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(val_accs, label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主训练函数"""
    # 加载配置
    config = load_config()
    print("📋 训练配置:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    # 设置设备
    device = setup_device()
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 初始化wandb（可选）
    use_wandb = input("是否使用Wandb记录训练? (y/n): ").lower() == 'y'
    if use_wandb:
        wandb.init(project='ai-ml-pipeline', config=config)
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    
    # TODO: 创建数据加载器
    # train_loader = create_dataloader(config, 'train')
    # val_loader = create_dataloader(config, 'val')
    
    print("⚠️ 请在这里添加您的数据加载代码")
    print("💡 提示: 取消注释上面的代码并实现create_dataloader函数")
    
    # 训练循环
    epochs = config['training']['epochs']
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # TODO: 实际训练代码
        # train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 示例数据（请替换为实际训练结果）
        train_loss, train_acc = 0.5, 85.0
        val_loss, val_acc = 0.6, 82.0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        
        # 记录到wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, 'models/best_model.pth')
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, f'models/checkpoint_epoch_{epoch+1}.pth')
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print(f"\n🎉 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 