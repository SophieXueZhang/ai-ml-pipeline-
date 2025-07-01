#!/usr/bin/env python3
"""超简单同步脚本"""
import subprocess
import sys

msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "快速更新代码"

subprocess.run(f'git add . && git commit -m "{msg}" && git push origin main', shell=True)
print(f"✅ 已同步到GitHub！在Colab中运行热重载单元格即可获取最新代码") 