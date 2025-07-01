#!/usr/bin/env python3
"""
开发环境设置脚本
设置本地开发环境，创建必要的目录结构，配置Git钩子等
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def create_directory_structure():
    """创建项目目录结构"""
    directories = [
        'src',
        'src/models',
        'src/data',
        'src/training',
        'src/inference',
        'notebooks',
        'colab_templates',
        'configs',
        'data',
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'outputs',
        'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # 创建 __init__.py 文件用于Python包
        if directory.startswith('src') or directory == 'utils':
            init_file = Path(directory) / '__init__.py'
            if not init_file.exists():
                init_file.touch()
    
    print("✅ 目录结构创建完成")

def create_config_files():
    """创建配置文件"""
    
    # 创建默认训练配置
    train_config = {
        "model": {
            "name": "resnet50",
            "pretrained": True,
            "num_classes": 10
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "optimizer": "adam"
        },
        "data": {
            "train_path": "data/processed/train",
            "val_path": "data/processed/val",
            "test_path": "data/processed/test"
        },
        "colab": {
            "use_gpu": True,
            "mount_drive": True,
            "sync_interval": 300
        }
    }
    
    with open('configs/train_config.json', 'w', encoding='utf-8') as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    
    # 创建Colab配置
    colab_config = {
        "sync": {
            "github_repo": "",
            "branch": "main",
            "local_path": "/content/ai-ml-pipeline",
            "exclude_patterns": [".git", "__pycache__", "*.pyc", ".DS_Store"]
        },
        "runtime": {
            "gpu_type": "T4",
            "runtime_type": "gpu"
        }
    }
    
    with open('configs/colab_config.json', 'w', encoding='utf-8') as f:
        json.dump(colab_config, f, indent=2, ensure_ascii=False)
    
    print("✅ 配置文件创建完成")

def create_gitignore():
    """创建.gitignore文件"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv/

# Jupyter Notebook
.ipynb_checkpoints

# 数据文件
data/raw/
data/processed/
*.csv
*.json
*.pkl
*.h5

# 模型文件
models/*.pth
models/*.pt
models/*.ckpt
*.model

# 日志文件
logs/
*.log
wandb/

# 输出文件
outputs/
results/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db

# 环境变量
.env
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore 文件创建完成")

def check_dependencies():
    """检查Python依赖"""
    try:
        import pip
        print("✅ pip 可用")
        
        # 检查是否已安装requirements.txt中的包
        print("📦 检查依赖包...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        installed_packages = result.stdout.lower()
        
        required_packages = ['numpy', 'pandas', 'torch', 'transformers']
        missing_packages = []
        
        for package in required_packages:
            if package not in installed_packages:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"⚠️  缺少以下包: {', '.join(missing_packages)}")
            print("💡 运行: pip install -r requirements.txt")
        else:
            print("✅ 主要依赖包已安装")
            
    except ImportError:
        print("❌ pip 不可用")

def main():
    """主函数"""
    print("🚀 开始设置AI ML Pipeline开发环境...")
    print("=" * 50)
    
    # 创建目录结构
    create_directory_structure()
    
    # 创建配置文件
    create_config_files()
    
    # 创建.gitignore
    create_gitignore()
    
    # 检查依赖
    check_dependencies()
    
    print("=" * 50)
    print("🎉 开发环境设置完成!")
    print("\n📝 下一步:")
    print("1. 运行: pip install -r requirements.txt")
    print("2. 打开 colab_templates/colab_sync.ipynb")
    print("3. 开始在 src/ 目录下编写代码")
    print("4. 使用Colab进行GPU训练")

if __name__ == "__main__":
    main() 