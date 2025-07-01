#!/usr/bin/env python3
"""
AI ML Pipeline 简化演示脚本
展示项目结构和基本工作流程（不依赖外部ML包）
"""

import os
import json
from pathlib import Path

def show_project_structure():
    """显示项目结构"""
    print("📁 AI ML Pipeline 项目结构:")
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
                
                if os.path.isdir(path) and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    print_tree(path, prefix + extension, max_depth, current_depth + 1)
        except PermissionError:
            pass
    
    print_tree(".")

def show_configuration():
    """显示配置信息"""
    print("\n⚙️ 配置文件内容:")
    print("=" * 50)
    
    # 显示训练配置
    if os.path.exists('configs/train_config.json'):
        print("📋 训练配置 (train_config.json):")
        with open('configs/train_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(json.dumps(config, indent=2, ensure_ascii=False))
    
    print("\n" + "-" * 30)
    
    # 显示Colab配置
    if os.path.exists('configs/colab_config.json'):
        print("🔧 Colab配置 (colab_config.json):")
        with open('configs/colab_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(json.dumps(config, indent=2, ensure_ascii=False))

def show_file_stats():
    """显示文件统计"""
    print("\n📊 项目文件统计:")
    print("=" * 50)
    
    file_count = 0
    dir_count = 0
    total_size = 0
    
    for root, dirs, files in os.walk('.'):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        dir_count += len(dirs)
        for file in files:
            if not file.startswith('.'):
                file_count += 1
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    pass
    
    print(f"📁 目录数量: {dir_count}")
    print(f"📄 文件数量: {file_count}")
    print(f"💾 总大小: {total_size / 1024:.1f} KB")

def show_key_files():
    """显示关键文件说明"""
    print("\n🔑 关键文件说明:")
    print("=" * 50)
    
    key_files = {
        "README.md": "项目主要说明文档",
        "快速开始指南.md": "详细的使用指南",
        "requirements.txt": "Python依赖包列表",
        "setup_dev_env.py": "开发环境设置脚本",
        "demo.py": "完整功能演示脚本",
        "colab_templates/colab_sync.ipynb": "Colab同步和训练笔记本",
        "src/training/train.py": "训练脚本模板",
        "src/models/example_model.py": "示例模型定义",
        "utils/colab_utils.py": "Colab集成工具函数",
        "configs/train_config.json": "训练配置文件",
        "configs/colab_config.json": "Colab配置文件"
    }
    
    for file_path, description in key_files.items():
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"{status} {file_path:<35} - {description}")

def show_workflow():
    """显示工作流程"""
    print("\n🚀 完整工作流程:")
    print("=" * 50)
    
    workflow_steps = [
        "1. 本地开发阶段:",
        "   • 在 src/ 目录下编写ML代码",
        "   • 在 configs/ 目录下配置训练参数",
        "   • 本地测试和调试（小规模数据）",
        "",
        "2. 代码同步阶段:",
        "   • git add . && git commit -m '更新代码'",
        "   • git push origin main",
        "",
        "3. Colab训练阶段:",
        "   • 上传 colab_templates/colab_sync.ipynb 到Colab",
        "   • 设置运行时为GPU (T4/A100)",
        "   • 修改GitHub仓库URL",
        "   • 运行所有单元格",
        "",
        "4. 结果保存阶段:",
        "   • 模型自动保存到Google Drive",
        "   • 训练日志和结果同步",
        "   • 可选：使用Wandb跟踪实验"
    ]
    
    for step in workflow_steps:
        print(step)

def main():
    """主函数"""
    print("🎯 AI ML Pipeline 项目演示")
    print("🔧 本地开发 + Colab GPU 训练解决方案")
    print("=" * 60)
    
    # 显示项目结构
    show_project_structure()
    
    # 显示配置信息
    show_configuration()
    
    # 显示文件统计
    show_file_stats()
    
    # 显示关键文件
    show_key_files()
    
    # 显示工作流程
    show_workflow()
    
    print("\n" + "=" * 60)
    print("🎉 项目设置完成！您可以开始开发了")
    print("\n📖 接下来的步骤:")
    print("1. 阅读 '快速开始指南.md' 了解详细使用方法")
    print("2. 安装Python依赖: pip install -r requirements.txt")
    print("3. 在 src/ 目录下编写您的ML代码")
    print("4. 将代码推送到GitHub")
    print("5. 在Colab中使用 colab_sync.ipynb 进行GPU训练")
    print("\n💡 如需完整功能演示，请先安装依赖后运行 python demo.py")

if __name__ == "__main__":
    main() 