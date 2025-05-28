import os
import datetime
import glob  # 用于文件名模式匹配

# --- 配置区域 ---

# 1. 指定要搜索的目录
target_directory = "BDMAP_O_tensorf_200"  # 请替换成您要检查的实际目录路径

# 2. 指定文件名模式 (使用 glob 模式)
#    例如:
#    '*.log'       匹配所有以 .log 结尾的文件
#    'data_*.csv'  匹配所有以 data_ 开头、.csv 结尾的文件
#    'report.txt'  匹配名为 report.txt 的特定文件
filename_pattern = "pred.nii.gz" # 请替换成您的文件名模式

# 3. 指定截止日期 (YYYY-MM-DD 格式)
#    早于这一天的文件将被删除 (不包括这一天)
cutoff_date_str = "2025-04-21" # 请替换成您的截止日期

# 4. 是否执行 "演练" 模式 (Dry Run)
#    设置为 True 时，脚本只会打印哪些文件会被删除，但不会实际删除。
#    强烈建议先设置为 True 运行一次，确认无误后再设置为 False 执行删除。
dry_run = False

# --- 脚本执行区域 ---

print(f"--- 文件清理脚本 ---")
print(f"目标目录: {target_directory}")
print(f"文件名模式: {filename_pattern}")
print(f"删除修改日期早于 {cutoff_date_str} 的文件")
print(f"演练模式 (Dry Run): {'是' if dry_run else '否'}")
print("-" * 20)

# 检查目标目录是否存在
if not os.path.isdir(target_directory):
    print(f"错误: 目录 '{target_directory}' 不存在或不是一个有效的目录。")
    exit(1)

# 将截止日期字符串转换为 datetime 对象，然后获取其时间戳
try:
    # 我们比较的是时间戳，截止日期设为当天 00:00:00
    cutoff_date = datetime.datetime.strptime(cutoff_date_str, "%Y-%m-%d")
    cutoff_timestamp = cutoff_date.timestamp()
    print(f"截止时间戳: {cutoff_timestamp} (对应日期 {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')})")
except ValueError:
    print(f"错误: 截止日期格式无效 '{cutoff_date_str}'。请使用 YYYY-MM-DD 格式。")
    exit(1)

# 构建完整的文件搜索路径模式
search_pattern = os.path.join(target_directory, "*", filename_pattern)
print(f"正在搜索文件: {search_pattern}")

# 使用 glob 查找所有匹配的文件
matching_files = glob.glob(search_pattern)

deleted_count = 0
checked_count = len(matching_files)

if not matching_files:
    print("没有找到符合模式的文件。")
else:
    print(f"找到 {checked_count} 个匹配的文件，正在检查修改日期...")
    for filepath in matching_files:
        try:
            # 获取文件的状态信息
            file_stat = os.stat(filepath)
            # 获取最后修改时间戳 (对应 ls -l 显示的时间)
            mod_time_timestamp = file_stat.st_mtime
            mod_time_dt = datetime.datetime.fromtimestamp(mod_time_timestamp)

            # 比较文件的修改时间戳和截止时间戳
            if mod_time_timestamp < cutoff_timestamp:
                print(f"找到旧文件: {os.path.basename(filepath)} (修改于: {mod_time_dt.strftime('%Y-%m-%d %H:%M:%S')})")
                if not dry_run:
                    try:
                        os.remove(filepath)
                        print(f"  -> 已删除: {filepath}")
                        deleted_count += 1
                    except OSError as e:
                        print(f"  -> 删除失败: {filepath} - 错误: {e}")
                else:
                    print(f"  -> [演练模式] 将删除: {filepath}")
                    # 在演练模式下也增加计数器，以便报告将删除多少文件
                    deleted_count += 1
            # else:
                # 如果需要，可以取消注释下面这行来查看哪些文件被保留了
                # print(f"保留文件: {os.path.basename(filepath)} (修改于: {mod_time_dt.strftime('%Y-%m-%d %H:%M:%S')})")

        except FileNotFoundError:
            # 文件在 glob 找到后、os.stat 执行前被删除了
            print(f"警告: 文件在处理期间未找到: {filepath}")
        except PermissionError:
            print(f"警告: 没有权限访问文件: {filepath}")
        except Exception as e:
            print(f"处理文件时发生意外错误 {filepath}: {e}")

print("-" * 20)
if dry_run:
    print(f"演练完成。共检查 {checked_count} 个文件，其中 {deleted_count} 个文件的修改日期早于 {cutoff_date_str}，它们将在非演练模式下被删除。")
else:
    print(f"清理完成。共检查 {checked_count} 个文件，实际删除了 {deleted_count} 个文件。")
print("--- 脚本结束 ---")