#!/usr/bin/env python3
"""
文件清理工具
功能：
1. 检测重复文件
2. 检测长时间未使用的文件
3. 提供安全的清理选项
"""

import os
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from collections import defaultdict
import json
import shutil

class FileCleanerTool:
    def __init__(self, target_dir="."):
        self.target_dir = Path(target_dir).resolve()
        self.duplicate_files = defaultdict(list)
        self.old_files = []
        
    def calculate_file_hash(self, file_path):
        """计算文件的MD5哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # 分块读取，避免大文件占用过多内存
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"计算哈希值失败 {file_path}: {e}")
            return None
    
    def find_duplicate_files(self, file_extensions=None, min_size=1024):
        """
        查找重复文件
        file_extensions: 要检查的文件扩展名列表，None表示检查所有文件
        min_size: 最小文件大小（字节），避免检查空文件或很小的文件
        """
        print(f"正在扫描目录: {self.target_dir}")
        print(f"查找重复文件（最小大小: {min_size} 字节）...")
        
        file_hashes = defaultdict(list)
        
        for root, dirs, files in os.walk(self.target_dir):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # 跳过隐藏文件
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                
                # 检查文件扩展名
                if file_extensions and file_path.suffix.lower() not in file_extensions:
                    continue
                
                try:
                    # 检查文件大小
                    if file_path.stat().st_size < min_size:
                        continue
                    
                    # 计算哈希值
                    file_hash = self.calculate_file_hash(file_path)
                    if file_hash:
                        file_info = {
                            'path': str(file_path),
                            'size': file_path.stat().st_size,
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                        }
                        file_hashes[file_hash].append(file_info)
                        
                except Exception as e:
                    print(f"处理文件失败 {file_path}: {e}")
        
        # 找出重复文件
        self.duplicate_files = {hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1}
        
        return self.duplicate_files
    
    def find_old_files(self, days=30, file_extensions=None):
        """
        查找长时间未使用的文件
        days: 多少天未修改算作旧文件
        file_extensions: 要检查的文件扩展名列表
        """
        print(f"查找 {days} 天内未修改的文件...")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        self.old_files = []
        
        for root, dirs, files in os.walk(self.target_dir):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # 跳过隐藏文件
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                
                # 检查文件扩展名
                if file_extensions and file_path.suffix.lower() not in file_extensions:
                    continue
                
                try:
                    stat = file_path.stat()
                    # 使用最后修改时间和最后访问时间中的较新者
                    last_used = max(
                        datetime.fromtimestamp(stat.st_mtime),
                        datetime.fromtimestamp(stat.st_atime)
                    )
                    
                    if last_used < cutoff_date:
                        file_info = {
                            'path': str(file_path),
                            'size': stat.st_size,
                            'last_modified': datetime.fromtimestamp(stat.st_mtime),
                            'last_accessed': datetime.fromtimestamp(stat.st_atime),
                            'last_used': last_used
                        }
                        self.old_files.append(file_info)
                        
                except Exception as e:
                    print(f"处理文件失败 {file_path}: {e}")
        
        # 按最后使用时间排序
        self.old_files.sort(key=lambda x: x['last_used'])
        return self.old_files
    
    def format_size(self, size_bytes):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def print_duplicate_report(self):
        """打印重复文件报告"""
        if not self.duplicate_files:
            print("✅ 没有找到重复文件")
            return
        
        total_wasted_space = 0
        print(f"\n📋 找到 {len(self.duplicate_files)} 组重复文件：")
        print("=" * 80)
        
        for i, (hash_val, files) in enumerate(self.duplicate_files.items(), 1):
            print(f"\n组 {i} (MD5: {hash_val[:8]}...):")
            print(f"  文件大小: {self.format_size(files[0]['size'])}")
            print(f"  重复数量: {len(files)} 个文件")
            
            # 计算浪费的空间（保留一个原文件）
            wasted_space = files[0]['size'] * (len(files) - 1)
            total_wasted_space += wasted_space
            print(f"  浪费空间: {self.format_size(wasted_space)}")
            
            for j, file_info in enumerate(files):
                marker = "📁 [保留]" if j == 0 else "🗑️ [可删除]"
                print(f"    {marker} {file_info['path']}")
                print(f"        修改时间: {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n💾 总计可节省空间: {self.format_size(total_wasted_space)}")
    
    def print_old_files_report(self):
        """打印旧文件报告"""
        if not self.old_files:
            print("✅ 没有找到长时间未使用的文件")
            return
        
        total_size = sum(file['size'] for file in self.old_files)
        print(f"\n📋 找到 {len(self.old_files)} 个长时间未使用的文件：")
        print(f"💾 总大小: {self.format_size(total_size)}")
        print("=" * 80)
        
        for file_info in self.old_files:
            print(f"\n📄 {file_info['path']}")
            print(f"   大小: {self.format_size(file_info['size'])}")
            print(f"   最后修改: {file_info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   最后访问: {file_info['last_accessed'].strftime('%Y-%m-%d %H:%M:%S')}")
            days_ago = (datetime.now() - file_info['last_used']).days
            print(f"   {days_ago} 天未使用")
    
    def clean_duplicate_files(self, auto_clean=False):
        """
        清理重复文件
        auto_clean: 是否自动清理（保留最新的文件）
        """
        if not self.duplicate_files:
            print("没有重复文件需要清理")
            return
        
        cleaned_count = 0
        saved_space = 0
        
        for hash_val, files in self.duplicate_files.items():
            # 按修改时间排序，保留最新的文件
            files.sort(key=lambda x: x['modified'], reverse=True)
            files_to_delete = files[1:]  # 删除除了最新文件之外的所有文件
            
            print(f"\n处理重复文件组 (保留: {files[0]['path']}):")
            
            for file_info in files_to_delete:
                file_path = Path(file_info['path'])
                if not file_path.exists():
                    continue
                
                if auto_clean:
                    try:
                        file_path.unlink()
                        print(f"  ✅ 已删除: {file_path}")
                        cleaned_count += 1
                        saved_space += file_info['size']
                    except Exception as e:
                        print(f"  ❌ 删除失败: {file_path} - {e}")
                else:
                    response = input(f"  删除 {file_path}? [y/N/q]: ").strip().lower()
                    if response == 'q':
                        print("退出清理过程")
                        break
                    elif response == 'y':
                        try:
                            file_path.unlink()
                            print(f"  ✅ 已删除: {file_path}")
                            cleaned_count += 1
                            saved_space += file_info['size']
                        except Exception as e:
                            print(f"  ❌ 删除失败: {file_path} - {e}")
                    else:
                        print(f"  ⏭️ 跳过: {file_path}")
        
        print(f"\n🎉 清理完成!")
        print(f"📊 删除了 {cleaned_count} 个重复文件")
        print(f"💾 节省空间: {self.format_size(saved_space)}")
    
    def clean_old_files(self, auto_clean=False):
        """
        清理旧文件
        auto_clean: 是否自动清理
        """
        if not self.old_files:
            print("没有旧文件需要清理")
            return
        
        cleaned_count = 0
        saved_space = 0
        
        for file_info in self.old_files:
            file_path = Path(file_info['path'])
            if not file_path.exists():
                continue
            
            days_ago = (datetime.now() - file_info['last_used']).days
            
            if auto_clean:
                try:
                    file_path.unlink()
                    print(f"✅ 已删除: {file_path} ({days_ago}天未使用)")
                    cleaned_count += 1
                    saved_space += file_info['size']
                except Exception as e:
                    print(f"❌ 删除失败: {file_path} - {e}")
            else:
                print(f"\n📄 {file_path}")
                print(f"   大小: {self.format_size(file_info['size'])}, {days_ago}天未使用")
                response = input("  删除此文件? [y/N/q]: ").strip().lower()
                if response == 'q':
                    print("退出清理过程")
                    break
                elif response == 'y':
                    try:
                        file_path.unlink()
                        print(f"  ✅ 已删除: {file_path}")
                        cleaned_count += 1
                        saved_space += file_info['size']
                    except Exception as e:
                        print(f"  ❌ 删除失败: {file_path} - {e}")
                else:
                    print(f"  ⏭️ 跳过: {file_path}")
        
        print(f"\n🎉 清理完成!")
        print(f"📊 删除了 {cleaned_count} 个旧文件")
        print(f"💾 节省空间: {self.format_size(saved_space)}")
    
    def save_report(self, filename="file_cleanup_report.json"):
        """保存清理报告到JSON文件"""
        report = {
            'scan_time': datetime.now().isoformat(),
            'target_directory': str(self.target_dir),
            'duplicate_files': {
                'groups': len(self.duplicate_files),
                'total_files': sum(len(files) for files in self.duplicate_files.values()),
                'details': {}
            },
            'old_files': {
                'count': len(self.old_files),
                'total_size': sum(file['size'] for file in self.old_files),
                'details': []
            }
        }
        
        # 添加重复文件详情
        for i, (hash_val, files) in enumerate(self.duplicate_files.items()):
            report['duplicate_files']['details'][f'group_{i+1}'] = {
                'hash': hash_val,
                'file_size': files[0]['size'],
                'file_count': len(files),
                'files': [{'path': f['path'], 'modified': f['modified'].isoformat()} for f in files]
            }
        
        # 添加旧文件详情
        for file_info in self.old_files:
            report['old_files']['details'].append({
                'path': file_info['path'],
                'size': file_info['size'],
                'last_used': file_info['last_used'].isoformat(),
                'days_unused': (datetime.now() - file_info['last_used']).days
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 报告已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description="文件清理工具")
    parser.add_argument("directory", nargs="?", default=".", help="要清理的目录路径")
    parser.add_argument("--days", type=int, default=30, help="多少天未使用算作旧文件 (默认: 30)")
    parser.add_argument("--min-size", type=int, default=1024, help="检查重复文件的最小大小 (字节, 默认: 1024)")
    parser.add_argument("--extensions", nargs="+", help="只检查指定扩展名的文件 (例如: .ipynb .py .txt)")
    parser.add_argument("--auto-clean-duplicates", action="store_true", help="自动清理重复文件")
    parser.add_argument("--auto-clean-old", action="store_true", help="自动清理旧文件")
    parser.add_argument("--report-only", action="store_true", help="只生成报告，不执行清理")
    parser.add_argument("--save-report", help="保存报告到指定文件")
    
    args = parser.parse_args()
    
    # 创建清理工具实例
    cleaner = FileCleanerTool(args.directory)
    
    print("🧹 文件清理工具启动")
    print(f"📁 目标目录: {cleaner.target_dir}")
    print("=" * 80)
    
    # 查找重复文件
    print("🔍 步骤 1: 查找重复文件...")
    duplicate_files = cleaner.find_duplicate_files(
        file_extensions=args.extensions,
        min_size=args.min_size
    )
    cleaner.print_duplicate_report()
    
    # 查找旧文件
    print(f"\n🔍 步骤 2: 查找 {args.days} 天内未使用的文件...")
    old_files = cleaner.find_old_files(
        days=args.days,
        file_extensions=args.extensions
    )
    cleaner.print_old_files_report()
    
    # 保存报告
    if args.save_report:
        cleaner.save_report(args.save_report)
    
    # 如果只是生成报告，则退出
    if args.report_only:
        print("\n📋 仅生成报告模式，不执行清理操作")
        return
    
    # 清理重复文件
    if duplicate_files:
        print("\n🧹 步骤 3: 清理重复文件...")
        if not args.auto_clean_duplicates:
            response = input("是否开始清理重复文件? [y/N]: ").strip().lower()
            if response != 'y':
                print("跳过重复文件清理")
            else:
                cleaner.clean_duplicate_files(auto_clean=False)
        else:
            cleaner.clean_duplicate_files(auto_clean=True)
    
    # 清理旧文件
    if old_files:
        print("\n🧹 步骤 4: 清理旧文件...")
        if not args.auto_clean_old:
            response = input("是否开始清理旧文件? [y/N]: ").strip().lower()
            if response != 'y':
                print("跳过旧文件清理")
            else:
                cleaner.clean_old_files(auto_clean=False)
        else:
            cleaner.clean_old_files(auto_clean=True)
    
    print("\n✨ 清理完成!")


if __name__ == "__main__":
    main() 