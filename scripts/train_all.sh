#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <data_root> <config> <output_root> [--dryrun] [-- <extra_args>]

Arguments:
  data_root    数据目录 或 glob 匹配模式（如 data/*LCA）
  config       配置文件路径
  output_root  输出根目录

Options:
  --dryrun     仅打印命令，不执行
  --           分隔符，之后的所有参数透传给 gs-fit (pixi run gs-fit -- ...)
                 示例: -- --logger wandb --trainer.max_steps 10000
  -h, --help   显示帮助信息
EOF
    exit 0
}

# --- 解析参数 ---
DRYRUN=false
PASSTHROUGH=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dryrun) DRYRUN=true; shift ;;
        --) shift; PASSTHROUGH=("$@"); break ;;
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

# --- 收集所有 case（支持 glob 通配符） ---
shopt -s nullglob
RAW_MATCHES=($DATA_ROOT)
shopt -u nullglob

if [[ ${#RAW_MATCHES[@]} -eq 0 ]]; then
    echo "Error: 没有匹配到任何路径: $DATA_ROOT"
    exit 1
fi

# 判断是否包含 glob 通配符
if [[ "$DATA_ROOT" == *[\*\?\[]* ]]; then
    # glob 模式：直接过滤匹配到的目录/符号链接
    CASES=()
    for match in "${RAW_MATCHES[@]}"; do
        if [[ -d "$match" || -L "$match" ]]; then
            CASES+=("$match")
        fi
    done
else
    # 普通目录：列出其下所有子目录/符号链接
    dir_path="${DATA_ROOT%/}"
    CASES=()
    while IFS= read -r item; do
        CASES+=("$item")
    done < <(find "$dir_path" -maxdepth 1 \( -type d -o -type l \) ! -name "$(basename "$dir_path")" | sort)
fi

TOTAL=${#CASES[@]}
echo "Found $TOTAL cases."

# 显示前 5 个 case 做预览
for ((i = 0; i < TOTAL && i < 5; i++)); do
    echo "  $(basename "${CASES[$i]}")"
done

# --- 计算实验名称（取第一个 case 的父目录名 + config 文件主名） ---
first_case="${CASES[0]}"
parent_dir=$(dirname "$first_case")
data_basename=$(basename "$parent_dir")
config_stem=$(basename "$CONFIG" | sed 's/\.[^.]*$//')
EXP_NAME="${data_basename}_${config_stem}"

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

    ALL_ARGS=(
        --output "$OUTPUT_ROOT"
        --config "$CONFIG"
        --data.path "$d"
        -n "$EXP_NAME"
        -v "$case_name"
        "${PASSTHROUGH[@]}"
    )

    if [[ "$DRYRUN" == true ]]; then
        echo "pixi run gs-fit -- ${ALL_ARGS[*]}"
        continue
    fi

    # 执行训练，失败时记录到 stderr 并继续下一个 case
    set +e
    pixi run gs-fit -- "${ALL_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        >&2 echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $case_name (exit code: $EXIT_CODE)"
        >&2 echo "    config: $CONFIG"
        >&2 echo "    data:   $d"
        >&2 echo "    output: $output_case_dir"
        >&2 echo "    cmd:    pixi run gs-fit -- ${ALL_ARGS[*]}"
        echo "[$idx/$TOTAL] FAILED $case_name (exit code: $EXIT_CODE)"
    fi
done
