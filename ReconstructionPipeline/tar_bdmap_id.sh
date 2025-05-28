#!/usr/bin/env bash

# 用法示例：
#   ./tar_subdirs.sh BDMAP_0000004

if [ $# -ne 1 ]; then
  echo "Usage: $0 <subdir_name>"
  echo "Example: $0 BDMAP_0000004"
  exit 1
fi

SUBDIR="$1"
OUTPUT="${SUBDIR}.tar.gz"

# 在当前目录下寻找所有以 BDMAP_O 开头的一级目录
BASE_DIRS=(BDMAP_O*)

# 收集所有存在该子目录的路径
FOUND_PATHS=()
for D in "${BASE_DIRS[@]}"; do
  if [ -d "$D/$SUBDIR" ]; then
    FOUND_PATHS+=("$D/$SUBDIR")
  else
    echo "Warning: 目录 '$D' 下未找到子目录 '$SUBDIR'"
  fi
done

if [ ${#FOUND_PATHS[@]} -eq 0 ]; then
  echo "Error: 未找到任何匹配的子目录 '$SUBDIR'"
  exit 1
fi

# 打包并压缩
tar czf "$OUTPUT" "${FOUND_PATHS[@]}"

# 列出打包结果
echo "已生成打包文件：$OUTPUT"
echo "打包内容如下："
for P in "${FOUND_PATHS[@]}"; do
  echo "  - $P"
done