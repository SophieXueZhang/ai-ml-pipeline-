#!/usr/bin/env python3
"""
快速开始 - 文件清理工具演示
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"💻 命令: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def main():
    print("🚀 文件清理工具快速演示")
    print("这个演示将展示如何使用文件清理工具")
    
    # 检查文件是否存在
    if not Path("file_cleaner.py").exists():
        print("❌ file_cleaner.py 文件不存在")
        return
    
    if not Path("clean_colab_files.py").exists():
        print("❌ clean_colab_files.py 文件不存在")
        return
    
    print("\n选择要演示的功能:")
    print("1. 📊 生成当前目录的清理报告")
    print("2. 🔍 检查特定类型文件(.ipynb, .py)")
    print("3. 📋 显示工具帮助信息")
    print("4. 🧹 运行简化版colab清理工具")
    print("5. ⚙️  查看所有可用参数")
    
    try:
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            # 生成报告
            cmd = "python file_cleaner.py --report-only --save-report demo_report.json"
            if run_command(cmd, "生成当前目录的清理报告"):
                print("\n📄 报告已保存到 demo_report.json")
                print("你可以查看这个文件了解详细信息")
        
        elif choice == '2':
            # 检查特定文件类型
            cmd = "python file_cleaner.py --report-only --extensions .ipynb .py .json"
            run_command(cmd, "检查.ipynb, .py, .json文件")
        
        elif choice == '3':
            # 显示帮助
            cmd = "python file_cleaner.py --help"
            run_command(cmd, "显示详细帮助信息")
        
        elif choice == '4':
            # 运行简化版工具（但不实际执行清理）
            print("\n🔧 简化版colab清理工具")
            print("💡 提示: 这个工具会自动找到包含.ipynb文件的目录")
            print("📝 实际使用时请运行: python clean_colab_files.py")
            
            # 展示会找到的文件
            current_dir = Path.cwd()
            ipynb_files = list(current_dir.glob("**/*.ipynb"))
            if ipynb_files:
                print(f"\n🔍 当前目录下找到 {len(ipynb_files)} 个.ipynb文件:")
                for file in ipynb_files[:5]:  # 只显示前5个
                    print(f"  📄 {file.relative_to(current_dir)}")
                if len(ipynb_files) > 5:
                    print(f"  ... 还有 {len(ipynb_files) - 5} 个文件")
            else:
                print("📝 当前目录下没有找到.ipynb文件")
        
        elif choice == '5':
            # 显示所有参数
            print("\n⚙️  通用清理工具参数:")
            params = [
                "python file_cleaner.py [目录路径]",
                "  --days 30              # 多少天算旧文件",
                "  --min-size 1024        # 最小文件大小",
                "  --extensions .py .txt  # 指定文件类型",
                "  --report-only          # 只生成报告",
                "  --auto-clean-duplicates # 自动清理重复文件",
                "  --save-report file.json # 保存报告"
            ]
            for param in params:
                print(param)
                
            print("\n🔧 简化版colab工具:")
            print("python clean_colab_files.py  # 直接运行即可")
        
        else:
            print("❌ 无效选择")
            return
    
    except KeyboardInterrupt:
        print("\n\n👋 演示结束")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    
    print("\n📚 更多使用方法请查看 README_清理工具使用说明.md")

if __name__ == "__main__":
    main() 