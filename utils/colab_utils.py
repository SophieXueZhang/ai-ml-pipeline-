#!/usr/bin/env python3
"""
Colab集成工具函数
提供本地开发与Colab GPU环境之间的桥接功能
"""

import os
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
import requests
from typing import Dict, List, Optional

class ColabSync:
    """Colab代码同步工具"""
    
    def __init__(self, config_path: str = 'configs/colab_config.json'):
        self.config = self.load_config(config_path)
        self.local_path = self.config.get('sync', {}).get('local_path', '/content/ai-ml-pipeline')
        
    def load_config(self, config_path: str) -> Dict:
        """加载Colab配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def setup_colab_environment(self):
        """设置Colab环境"""
        print("🚀 设置Colab环境...")
        
        # 检查GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
                print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("⚠️ GPU不可用")
        except ImportError:
            print("⚠️ PyTorch未安装")
        
        # 挂载Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive已挂载")
        except ImportError:
            print("⚠️ 不在Colab环境中")
        
        # 创建工作目录
        os.makedirs(self.local_path, exist_ok=True)
        os.chdir(self.local_path)
        print(f"✅ 工作目录: {os.getcwd()}")
    
    def sync_from_github(self, repo_url: str, branch: str = 'main'):
        """从GitHub同步代码"""
        print(f"📥 从GitHub同步代码: {repo_url}")
        
        if os.path.exists('.git'):
            # 更新现有仓库
            result = subprocess.run(['git', 'pull', 'origin', branch], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ 代码已更新")
            else:
                print(f"⚠️ 更新失败: {result.stderr}")
        else:
            # 克隆新仓库
            result = subprocess.run(['git', 'clone', repo_url, '.'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ 代码已克隆")
            else:
                print(f"⚠️ 克隆失败: {result.stderr}")
    
    def upload_from_local(self, zip_path: str):
        """从本地上传zip文件"""
        print(f"📁 解压文件: {zip_path}")
        
        if not os.path.exists(zip_path):
            print(f"⚠️ 文件不存在: {zip_path}")
            return
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        print("✅ 文件已解压")
        os.remove(zip_path)  # 删除zip文件
    
    def install_requirements(self, requirements_path: str = 'requirements.txt'):
        """安装项目依赖"""
        print("📦 安装项目依赖...")
        
        if os.path.exists(requirements_path):
            result = subprocess.run(['pip', 'install', '-r', requirements_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ 依赖安装完成")
            else:
                print(f"⚠️ 安装失败: {result.stderr}")
        else:
            # 安装基础依赖
            basic_packages = [
                'torch', 'torchvision', 'torchaudio',
                'transformers', 'datasets', 'wandb',
                'matplotlib', 'seaborn', 'tqdm'
            ]
            for package in basic_packages:
                subprocess.run(['pip', 'install', package], 
                             capture_output=True, text=True)
            print("✅ 基础依赖安装完成")
    
    def setup_python_path(self):
        """设置Python路径"""
        import sys
        paths_to_add = [
            os.path.join(self.local_path, 'src'),
            os.path.join(self.local_path, 'utils')
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.append(path)
        
        print("✅ Python路径已设置")
    
    def save_to_drive(self, source_dir: str, drive_path: str = '/content/drive/MyDrive/ai-ml-results'):
        """保存结果到Google Drive"""
        print(f"💾 保存到Drive: {drive_path}")
        
        os.makedirs(drive_path, exist_ok=True)
        
        if os.path.exists(source_dir):
            if os.path.isfile(source_dir):
                shutil.copy2(source_dir, drive_path)
            else:
                dest_dir = os.path.join(drive_path, os.path.basename(source_dir))
                shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
            print(f"✅ 已保存: {source_dir} -> {drive_path}")
        else:
            print(f"⚠️ 源目录不存在: {source_dir}")

class ColabMonitor:
    """Colab资源监控工具"""
    
    @staticmethod
    def check_gpu_memory():
        """检查GPU内存使用情况"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"GPU内存使用情况:")
                print(f"  已分配: {allocated:.2f} GB")
                print(f"  已保留: {reserved:.2f} GB") 
                print(f"  总内存: {total:.2f} GB")
                print(f"  剩余: {total - reserved:.2f} GB")
                
                return {
                    'allocated': allocated,
                    'reserved': reserved,
                    'total': total,
                    'free': total - reserved
                }
            else:
                print("⚠️ GPU不可用")
                return None
        except ImportError:
            print("⚠️ PyTorch未安装")
            return None
    
    @staticmethod
    def check_disk_usage():
        """检查磁盘使用情况"""
        result = subprocess.run(['df', '-h', '/content'], 
                              capture_output=True, text=True)
        print("磁盘使用情况:")
        print(result.stdout)
    
    @staticmethod
    def nvidia_smi():
        """运行nvidia-smi命令"""
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU信息:")
            print(result.stdout)
        else:
            print("⚠️ nvidia-smi不可用")

def quick_setup(github_repo: Optional[str] = None):
    """快速设置Colab环境"""
    print("🚀 快速设置Colab环境...")
    
    sync = ColabSync()
    
    # 设置环境
    sync.setup_colab_environment()
    
    # 同步代码
    if github_repo:
        sync.sync_from_github(github_repo)
    else:
        print("💡 提示: 请手动上传代码或提供GitHub仓库URL")
    
    # 安装依赖
    sync.install_requirements()
    
    # 设置Python路径
    sync.setup_python_path()
    
    print("✅ 环境设置完成！")
    
    # 显示系统信息
    monitor = ColabMonitor()
    monitor.check_gpu_memory()
    monitor.check_disk_usage()
    
    return sync

# 便捷函数
def setup_wandb(project_name: str = 'ai-ml-pipeline'):
    """设置Weights & Biases"""
    try:
        import wandb
        wandb.login()
        wandb.init(project=project_name)
        print("✅ Wandb已初始化")
        return True
    except Exception as e:
        print(f"⚠️ Wandb设置失败: {e}")
        return False

def download_from_url(url: str, filename: str):
    """从URL下载文件"""
    print(f"📥 下载文件: {filename}")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 下载完成: {filename}")
    else:
        print(f"⚠️ 下载失败: {response.status_code}")

def create_colab_notebook(template_name: str = 'basic_training'):
    """创建Colab笔记本模板"""
    templates = {
        'basic_training': {
            'cells': [
                {
                    'cell_type': 'markdown',
                    'source': ['# AI ML Pipeline - 基础训练模板\n']
                },
                {
                    'cell_type': 'code',
                    'source': [
                        '# 快速设置环境\n',
                        'from utils.colab_utils import quick_setup\n',
                        'sync = quick_setup("your-github-repo")\n'
                    ]
                },
                {
                    'cell_type': 'code',
                    'source': [
                        '# 开始训练\n',
                        'from src.training.train import main\n',
                        'main()\n'
                    ]
                }
            ]
        }
    }
    
    if template_name in templates:
        notebook_path = f'colab_templates/{template_name}.ipynb'
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump({
                'cells': templates[template_name]['cells'],
                'metadata': {
                    'colab': {'provenance': []},
                    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                    'language_info': {'name': 'python'}
                },
                'nbformat': 4,
                'nbformat_minor': 0
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 笔记本模板已创建: {notebook_path}")
    else:
        print(f"⚠️ 未知模板: {template_name}") 