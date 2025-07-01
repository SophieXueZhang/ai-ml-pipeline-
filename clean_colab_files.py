#!/usr/bin/env python3
"""
Colab文件清理工具
专门用于清理Jupyter Notebook和相关文件
"""

import os
import sys
from pathlib import Path

# 导入主清理工具
from file_cleaner import FileCleanerTool

def clean_colab_files():
    """清理colab相关文件的简化接口"""
    
    print("🔧 Colab文件清理工具")
    print("=" * 50)
    
    # 获取当前目录
    current_dir = Path.cwd()
    print(f"📁 当前目录: {current_dir}")
    
    # 查找可能的colab目录
    colab_dirs = []
    for item in current_dir.iterdir():
        if item.is_dir():
            item_name_lower = item.name.lower()
            if any(keyword in item_name_lower for keyword in ['colab', 'notebook', 'jupyter', 'ipynb']):
                colab_dirs.append(item)
    
    # 如果没找到特定的colab目录，检查当前目录是否有.ipynb文件
    ipynb_files = list(current_dir.glob("*.ipynb"))
    if ipynb_files and not colab_dirs:
        colab_dirs.append(current_dir)
    
    if colab_dirs:
        print(f"🔍 找到以下可能的colab目录:")
        for i, dir_path in enumerate(colab_dirs, 1):
            ipynb_count = len(list(dir_path.glob("**/*.ipynb")))
            print(f"  {i}. {dir_path.name} ({ipynb_count} 个 .ipynb 文件)")
        
        # 让用户选择目录
        if len(colab_dirs) > 1:
            while True:
                try:
                    choice = input(f"\n请选择要清理的目录 (1-{len(colab_dirs)}, 或按Enter清理全部): ").strip()
                    if not choice:
                        # 清理所有目录
                        target_dirs = colab_dirs
                        break
                    else:
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(colab_dirs):
                            target_dirs = [colab_dirs[choice_num - 1]]
                            break
                        else:
                            print("❌ 无效选择，请重新输入")
                except ValueError:
                    print("❌ 请输入有效数字")
        else:
            target_dirs = colab_dirs
    else:
        print("⚠️  没有找到明显的colab目录，将在当前目录查找")
        target_dirs = [current_dir]
    
    # 针对每个目录进行清理
    for target_dir in target_dirs:
        print(f"\n🧹 清理目录: {target_dir}")
        print("-" * 50)
        
        # 创建清理工具实例
        cleaner = FileCleanerTool(target_dir)
        
        # 专门针对notebook相关文件的扩展名
        notebook_extensions = ['.ipynb', '.py', '.json', '.txt', '.md', '.csv', '.pkl', '.model']
        
        print("🔍 步骤 1: 查找重复的notebook文件...")
        duplicate_files = cleaner.find_duplicate_files(
            file_extensions=notebook_extensions,
            min_size=512  # 降低最小文件大小，因为一些配置文件可能比较小
        )
        cleaner.print_duplicate_report()
        
        print(f"\n🔍 步骤 2: 查找30天内未使用的文件...")
        old_files = cleaner.find_old_files(
            days=30,
            file_extensions=notebook_extensions
        )
        cleaner.print_old_files_report()
        
        # 如果找到需要清理的文件，询问用户是否执行清理
        has_files_to_clean = bool(duplicate_files or old_files)
        
        if has_files_to_clean:
            print(f"\n📊 清理摘要:")
            if duplicate_files:
                total_duplicates = sum(len(files) - 1 for files in duplicate_files.values())
                print(f"  🔄 可删除的重复文件: {total_duplicates} 个")
            if old_files:
                print(f"  ⏰ 长时间未使用的文件: {len(old_files)} 个")
            
            # 询问是否执行清理
            print(f"\n❓ 是否开始清理 {target_dir.name} 目录？")
            print("  1. 只清理重复文件")
            print("  2. 只清理旧文件") 
            print("  3. 清理所有文件")
            print("  4. 跳过此目录")
            
            while True:
                choice = input("请选择 (1-4): ").strip()
                if choice == '1' and duplicate_files:
                    cleaner.clean_duplicate_files(auto_clean=False)
                    break
                elif choice == '2' and old_files:
                    cleaner.clean_old_files(auto_clean=False)
                    break
                elif choice == '3':
                    if duplicate_files:
                        cleaner.clean_duplicate_files(auto_clean=False)
                    if old_files:
                        cleaner.clean_old_files(auto_clean=False)
                    break
                elif choice == '4':
                    print(f"⏭️  跳过目录: {target_dir}")
                    break
                else:
                    print("❌ 无效选择，请重新输入")
        else:
            print("✅ 此目录不需要清理")
    
    print("\n🎉 所有清理任务完成!")
    
    # 提供一些建议
    print("\n💡 建议:")
    print("  • 定期运行此工具以保持文件整洁")
    print("  • 重要文件请备份后再清理")
    print("  • 可以使用 --report-only 参数先查看报告")

def main():
    try:
        clean_colab_files()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 