#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <data_root> <config> <output_root> [--wandb] [--dryrun]

Arguments:
  data_root    数据目录（包含各 case 子目录/符号链接）
  config       配置文件路径
  output_root  输出根目录

Options:
  --wandb      使用 wandb 记录日志
  --dryrun     仅打印命令，不执行
  -h, --help   显示帮助信息
EOF
    exit 0
}

# --- 解析参数 ---
WANDB=false
DRYRUN=false
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb) WANDB=true; shift ;;
        --dryrun) DRYRUN=true; shift ;;
        -h|--help) usage ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

# 恢复位置参数
set -- "${POSITIONAL[@]}"

if [[ $# -lt 3 ]]; then
    echo "Error: 缺少必要参数"
    usage
fi

DATA_ROOT="$1"
CONFIG="$2"
OUTPUT_ROOT="$3"

# --- 校验 ---
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: 数据目录 '$DATA_ROOT' 不存在."
    exit 1
fi
echo "Data root: $DATA_ROOT"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: 配置文件 '$CONFIG' 不存在."
    exit 1
fi
echo "Config: $CONFIG"

if [[ ! -d "$OUTPUT_ROOT" ]]; then
    echo "Output root '$OUTPUT_ROOT' 不存在，正在创建."
    mkdir -p "$OUTPUT_ROOT"
fi
echo "Output root: $OUTPUT_ROOT"

# --- 计算实验名称 ---
data_basename=$(basename "$DATA_ROOT")
config_stem=$(basename "$CONFIG" | sed 's/\.[^.]*$//')
EXP_NAME="${data_basename}_${config_stem}"

# --- 收集所有 case（目录或符号链接） ---
CASES=()
while IFS= read -r item; do
    CASES+=("$item")
done < <(find "$DATA_ROOT" -maxdepth 1 \( -type d -o -type l \) ! -name "$(basename "$DATA_ROOT")" | sort)

TOTAL=${#CASES[@]}
echo "Found $TOTAL cases in $DATA_ROOT."

# --- 逐个训练 ---
for ((i = 0; i < TOTAL; i++)); do
    idx=$((i + 1))
    d="${CASES[$i]}"
    case_name=$(basename "$d")

    output_case_dir="$OUTPUT_ROOT/$EXP_NAME/$case_name"

    if [[ -d "$output_case_dir" ]]; then
        echo "[$idx/$TOTAL] SKIP existing $case_name"
        continue
    fi

    echo "[$idx/$TOTAL] RUN  $case_name"

    CMD_BASE=(
        python main.py fit
        --output "$OUTPUT_ROOT"
        --config "$CONFIG"
        --data.path "$d"
        -n "$EXP_NAME"
        -v "$case_name"
    )

    if [[ "$DRYRUN" == true ]]; then
        if [[ "$WANDB" == true ]]; then
            echo "pixi run gs-fit -- ${CMD_BASE[*]} --logger wandb"
        else
            echo "${CMD_BASE[*]}"
        fi
        continue
    fi

    if [[ "$WANDB" == true ]]; then
        pixi run gs-fit -- "${CMD_BASE[@]}" --logger wandb
    else
        "${CMD_BASE[@]}"
    fi
done
