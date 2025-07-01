#!/usr/bin/env python3
"""
一键同步到Colab脚本
简化本地开发到Colab GPU训练的工作流程
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"📋 {description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description}成功")
        if result.stdout.strip():
            print(f"   输出: {result.stdout.strip()}")
    else:
        print(f"❌ {description}失败")
        print(f"   错误: {result.stderr.strip()}")
        return False
    return True

def main():
    """主函数"""
    print("🚀 AI ML Pipeline - 一键同步到Colab")
    print("=" * 50)
    
    # 检查是否有未提交的更改
    print("🔍 检查代码更改...")
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    
    if not result.stdout.strip():
        print("ℹ️  没有新的更改需要同步")
        print("💡 您可以直接在Colab中运行同步单元格")
        return
    
    print("📝 发现以下更改:")
    print(result.stdout)
    
    # 获取提交信息
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    default_message = f"更新ML代码 - {timestamp}"
    
    commit_message = input(f"💬 提交信息 (回车使用默认): ") or default_message
    
    # 执行Git操作
    commands = [
        ("git add .", "添加文件到暂存区"),
        (f'git commit -m "{commit_message}"', "提交更改"),
        ("git push origin main", "推送到GitHub")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return
    
    print("\n" + "=" * 50)
    print("🎉 同步完成！")
    print("\n📝 接下来在Colab中:")
    print("1. 运行GitHub同步单元格（重新从GitHub拉取代码）")
    print("2. 运行训练单元格")
    print("3. 享受GPU加速训练！")
    print("\n💡 您的GitHub仓库:")
    print("   https://github.com/SophieXueZhang/ai-ml-pipeline-.git")

if __name__ == "__main__":
    main() 