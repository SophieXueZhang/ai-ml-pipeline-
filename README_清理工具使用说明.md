# 文件清理工具使用说明

## 功能介绍

这个工具包含两个主要脚本：

1. **`file_cleaner.py`** - 通用文件清理工具
2. **`clean_colab_files.py`** - 专门用于清理Colab/Jupyter文件的简化版本

## 主要功能

✅ **重复文件检测** - 通过MD5哈希值检测完全相同的文件  
✅ **旧文件检测** - 找出长时间未使用的文件  
✅ **安全清理** - 交互式确认，避免误删重要文件  
✅ **智能保留** - 重复文件中自动保留最新的版本  
✅ **详细报告** - 显示文件大小、修改时间等详细信息  
✅ **空间统计** - 显示可节省的磁盘空间  

## 快速使用

### 方法1: 简化版Colab清理工具

```bash
# 直接运行简化版清理工具
python clean_colab_files.py
```

这个脚本会：
- 自动找到包含.ipynb文件的目录
- 专门检测notebook相关文件(.ipynb, .py, .json等)
- 提供友好的交互式界面

### 方法2: 通用文件清理工具

```bash
# 基本用法 - 清理当前目录
python file_cleaner.py

# 清理指定目录
python file_cleaner.py /path/to/directory

# 只检查特定类型的文件
python file_cleaner.py --extensions .ipynb .py .json

# 调整旧文件的天数阈值（默认30天）
python file_cleaner.py --days 60

# 只生成报告，不执行清理
python file_cleaner.py --report-only

# 自动清理重复文件（不询问）
python file_cleaner.py --auto-clean-duplicates

# 保存详细报告到文件
python file_cleaner.py --save-report cleanup_report.json
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `directory` | 要清理的目录路径 | 当前目录 |
| `--days` | 多少天未使用算作旧文件 | 30 |
| `--min-size` | 检查重复文件的最小大小(字节) | 1024 |
| `--extensions` | 只检查指定扩展名的文件 | 所有文件 |
| `--auto-clean-duplicates` | 自动清理重复文件 | 否 |
| `--auto-clean-old` | 自动清理旧文件 | 否 |
| `--report-only` | 只生成报告，不清理 | 否 |
| `--save-report` | 保存报告到指定文件 | 否 |

## 使用示例

### 示例1: 清理Jupyter Notebook文件

```bash
# 只检查和清理notebook相关文件
python file_cleaner.py --extensions .ipynb .py .json --days 45
```

### 示例2: 先查看报告再决定

```bash
# 先生成报告看看情况
python file_cleaner.py --report-only --save-report report.json

# 查看报告后再决定是否清理
python file_cleaner.py
```

### 示例3: 批量自动清理

```bash
# 自动清理重复文件，但手动确认旧文件
python file_cleaner.py --auto-clean-duplicates
```

## 安全提示

⚠️ **重要：使用前请备份重要文件！**

- 工具会在删除前显示详细信息
- 重复文件中会保留最新的版本
- 可以随时按 'q' 退出清理过程
- 建议先用 `--report-only` 查看会删除哪些文件

## 清理策略

### 重复文件处理
- 通过MD5哈希值识别完全相同的文件
- 自动保留修改时间最新的文件
- 删除其他重复副本

### 旧文件处理
- 基于最后修改时间和最后访问时间
- 使用较新的时间作为判断标准
- 可自定义天数阈值

## 支持的文件类型

工具支持所有文件类型，但针对Colab使用优化了以下类型：
- `.ipynb` - Jupyter Notebook文件
- `.py` - Python脚本
- `.json` - 配置文件
- `.txt` - 文本文件
- `.md` - Markdown文档
- `.csv` - 数据文件
- `.pkl` - Python pickle文件
- `.model` - 模型文件

## 故障排除

### 权限问题
如果遇到权限错误：
```bash
# macOS/Linux
chmod +x file_cleaner.py clean_colab_files.py

# Windows
# 确保以管理员身份运行
```

### 依赖问题
工具使用Python标准库，无需额外安装依赖。

### 中文显示问题
如果中文显示异常，确保终端支持UTF-8编码。

## 输出示例

```
🧹 文件清理工具启动
📁 目标目录: /Users/username/colab_files
================================================================================

🔍 步骤 1: 查找重复文件...
正在扫描目录: /Users/username/colab_files
查找重复文件（最小大小: 1024 字节）...

📋 找到 2 组重复文件：
================================================================================

组 1 (MD5: a1b2c3d4...):
  文件大小: 2.5 MB
  重复数量: 3 个文件
  浪费空间: 5.0 MB
    📁 [保留] /path/to/newest_file.ipynb
        修改时间: 2024-01-15 14:30:25
    🗑️ [可删除] /path/to/duplicate1.ipynb
        修改时间: 2024-01-10 09:15:30
    🗑️ [可删除] /path/to/duplicate2.ipynb
        修改时间: 2024-01-08 16:45:12

💾 总计可节省空间: 5.0 MB
``` 