#!/usr/bin/env python3
"""
AI ML Pipeline 演示脚本
展示完整的本地开发到Colab GPU训练工作流程
"""

import os
import json
import sys
from pathlib import Path

def check_environment():
    """检查环境设置"""
    print("🔍 检查环境设置...")
    
    # 检查目录结构
    required_dirs = ['src', 'configs', 'colab_templates', 'utils', 'data', 'models']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ 缺少目录: {missing_dirs}")
        print("💡 请运行: python setup_dev_env.py")
        return False
    else:
        print("✅ 目录结构完整")
    
    # 检查配置文件
    config_files = ['configs/train_config.json', 'configs/colab_config.json']
    missing_configs = []
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            missing_configs.append(config_file)
    
    if missing_configs:
        print(f"❌ 缺少配置文件: {missing_configs}")
        return False
    else:
        print("✅ 配置文件完整")
    
    # 检查Colab模板
    if not os.path.exists('colab_templates/colab_sync.ipynb'):
        print("❌ 缺少Colab同步笔记本")
        return False
    else:
        print("✅ Colab模板完整")
    
    return True

def demo_local_development():
    """演示本地开发流程"""
    print("\n📝 演示本地开发流程...")
    
    # 1. 加载配置
    print("1. 加载训练配置")
    with open('configs/train_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"   ✅ 配置加载完成: {config['model']['name']}")
    
    # 2. 导入模型
    print("2. 导入模型模块")
    try:
        sys.path.append('src')
        from models.example_model import create_model, model_summary
        print("   ✅ 模型模块导入成功")
        
        # 创建模型实例
        model = create_model(config['model'])
        print(f"   ✅ 模型创建成功: {model.__class__.__name__}")
        
    except ImportError as e:
        print(f"   ❌ 模型导入失败: {e}")
        return False
    
    # 3. 检查训练脚本
    print("3. 检查训练脚本")
    if os.path.exists('src/training/train.py'):
        print("   ✅ 训练脚本存在")
    else:
        print("   ❌ 训练脚本不存在")
    
    # 4. 检查工具函数
    print("4. 检查Colab工具")
    try:
        from utils.colab_utils import ColabSync, ColabMonitor
        print("   ✅ Colab工具可用")
    except ImportError as e:
        print(f"   ❌ Colab工具导入失败: {e}")
    
    return True

def show_colab_instructions():
    """显示Colab使用说明"""
    print("\n🚀 Colab使用说明:")
    print("=" * 50)
    
    instructions = [
        "1. 将代码推送到GitHub:",
        "   git add .",
        "   git commit -m '初始化项目'",
        "   git push origin main",
        "",
        "2. 打开Google Colab (https://colab.research.google.com/)",
        "",
        "3. 上传colab_templates/colab_sync.ipynb到Colab",
        "",
        "4. 在Colab中:",
        "   - 设置运行时为GPU (T4)",
        "   - 修改GITHUB_REPO变量为您的仓库URL",
        "   - 运行所有单元格",
        "",
        "5. 开始训练:",
        "   - 代码会自动同步",
        "   - GPU会被自动使用",
        "   - 结果会保存到Google Drive"
    ]
    
    for instruction in instructions:
        print(instruction)

def show_project_structure():
    """显示项目结构"""
    print("\n📁 项目结构:")
    print("=" * 50)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        try:
            items = sorted(os.listdir(directory))
            # 过滤隐藏文件和__pycache__
            items = [item for item in items if not item.startswith('.') and item != '__pycache__']
            
            for i, item in enumerate(items):
                path = os.path.join(directory, item)
                is_last = i == len(items) - 1
                
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item}")
                
                if os.path.isdir(path):
                    extension = "    " if is_last else "│   "
                    print_tree(path, prefix + extension, max_depth, current_depth + 1)
        except PermissionError:
            pass
    
    print_tree(".")

def main():
    """主函数"""
    print("🎯 AI ML Pipeline 演示")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请先运行设置脚本")
        return
    
    # 演示本地开发
    if not demo_local_development():
        print("\n❌ 本地开发演示失败")
        return
    
    # 显示项目结构
    show_project_structure()
    
    # 显示Colab说明
    show_colab_instructions()
    
    print("\n" + "=" * 50)
    print("🎉 演示完成！")
    print("\n💡 下一步:")
    print("1. 在src/目录下编写您的ML代码")
    print("2. 修改configs/目录下的配置文件")
    print("3. 将代码推送到GitHub")
    print("4. 在Colab中运行colab_sync.ipynb")
    print("5. 享受GPU加速训练！")

if __name__ == "__main__":
    main() 