#!/usr/bin/env nu
use std/log

def process_cases [
    cases: list<string>,
    config: path,
    output_root: path,
    extra: list<string>,
    apply: bool
] {
    let total = ($cases | length)

    print $"Found ($total) cases."
    print $"First 5 cases:"
    $cases | first 5 | print
    let config_name = $config | path parse | get stem

    for item in ($cases | enumerate) {
        let idx = ($item.index + 1)
        let d: string = $item.item

        if ($d | path type) == "File" {
            log warning $"Case ($d) is a file, expected a directory. Skipping."
            continue
        }

        let dataset_name = $d | path dirname | path basename
        let case_name = $d | path basename
        let exp_name = [$dataset_name, $config_name] | str join "_"

        let output_case_dir = (
            $output_root
            | path join $exp_name
            | path join $case_name
        )

        if ($output_case_dir | path exists) {
            print $"[($idx)/($total)] SKIP existing ($case_name)"
            continue
        }

        print $"[($idx)/($total)] RUN  ($case_name)"

        let all_args = [
            "--output", $output_root,
            "--config", $config,
            "--data.path", $d,
            "-n", $exp_name,
            "-v", $case_name,
            ...$extra
        ]

        let cmd = $"pixi run gs-fit -- ($all_args | str join ' ')"

        if not $apply {
            print $"DRYRUN: ($cmd)"
            continue
        }
        
        try {
            ^pixi run gs-fit -- ...$all_args
        } catch {|e|
            let log_line = [
                $"[(date now | format date '%Y-%m-%d %H:%M:%S')] FAILED: ($case_name)"
                $"    config: ($config)"
                $"    data:   ($d)"
                $"    output: ($output_case_dir)"
                $"    cmd:    ($cmd)"
                $"    error:  ($e.msg)"
                ""
            ] | str join "\n"

            log error $log_line
            if (($e.exit_code? | default 0) in [130 143]) {
                log error "Process was interrupted. Exiting."
                return $e.exit_code
            }
        }
    }
}

# 对输入命令进行训练，支持多种输入方式：单个 case、多个 case 列表、目录，并支持额外参数透传给 gs-fit，默认仅打印命令（dryrun）通过 --apply 参数实际执行训练，。
#
# 可以使用 `use scripts/train_all.nu` 在 nu shell 中加载模块，并 通过管道运算符传入所需 cases:
#   `ls data/gen_4d_output_all/flow/asoca-* | get name | train_all configs/gen_4d_output_all/flow-mlp.yaml outputs/RotCA-GR/ --apply --logger wandb --trainer.max_steps 5000`
#   `echo data/gen_4d_output_all/flow/asoca-* | train_all configs/gen_4d_output_all/flow-mlp.yaml outputs/RotCA-GR/`
#
# 也可以使用命令行方式调用脚本并直接指定输入（支持在bash中调用，但需要提前按照nushell）:
# 此时所有 --case, --case_list, --dir 参数识别到的cases将合并并去重，同时忽略管道输入
#   `scripts/train_all.nu configs/gen_4d_output_all/flow-mlp.yaml outputs/RotCA-GR/ --case data/gen_4d_output_all/flow/asoca-* --apply --logger wandb --trainer.max_steps 5000`
#       [注意] 如果是在 bash 中调用，glob 模式（如 asoca-*）需要用双引号包裹以防止被 bash 预先展开，应该写成 --case "data/gen_4d_output_all/flow/asoca-*"
#   `scripts/train_all.nu configs/gen_4d_output_all/flow-mlp.yaml outputs/RotCA-GR/ --case data/gen_4d_output_all/flow/asoca-normal__Normal_01__LCA`
#   `scripts/train_all.nu configs/gen_4d_output_all/flow-mlp.yaml outputs/RotCA-GR/ --dir data/gen_4d_output_all/flow`
#       [注意] 目前 case_list 不兼容 bash 调用
#
# 通过配置 `NU_LOG_LEVEL` 控制日志输出级别
export def --wrapped main [
    config: path,           # 配置文件路径
    output_root: path,      # 输出目录路径
    --case: glob          # 单个 case 的数据文件路径 或 glob pattern 字符串, 如: --case data/gen_4d_output_all/flow/asoca-*.
    --case_list: list<path>  # 多个 case 的数据文件路径列表, 如: --case_list 
    --dir: path             # cases 目录路径，目录下每个子目录为一个 case，子目录内包含数据文件, 如: --dir ./data/gen_4d_output_all/flow
    --apply                 # 是否实际执行训练（默认仅 dryrun 打印命令）
    ...extra: string        # 额外参数（列表），直接透传给 gs-fit
                            # 如: --logger wandb --trainer.max_steps 5000
]: [
    glob -> nothing       # 输入为单个数据文件路径或 glob 模式(等价于 --case)
    list<string> -> nothing  # 输入为数据文件路径列表(等价于 --case_list)
    nothing -> nothing      # 支持通过 --case 或 --dir 指定输入
] {
    let input = $in
    log debug $"input: ($input)"
    log debug $"config: ($config)"
    log debug $"output_root: ($output_root)"
    log debug $"case: ($case)"
    log debug $"case_list: ($case_list)"
    log debug $"dir: ($dir)"
    log debug $"apply: ($apply)"
    log debug $"extra: ($extra)"

    
    let cases = if ($case == null and $case_list == null and $dir == null) {
        if $input == null {
            log error "No input provided. Please provide --case, --case_list, --dir, or input data files."
            return
        }
        
        let $in_type = $input | describe
        log debug $"input type: ($in_type)"

        match $in_type {
            "string" => (glob $input)
            "list<string>" => $input
            _ => {
                log error $"Unsupported input type: ($in_type). Please provide a string or list of strings for input."
                []
            }
        }
    } else {
        if $input != null {
            log warning $"Input data files provided will be ignored because --case, --case_list, or --dir is also specified. input: ($input)"
        }

        let c1 = if ($case != null) { glob $case } else { [] }
        let c2 = if ($case_list != null) { $case_list } else { [] }
        let c3 = if ($dir != null) {
            print $"Listing cases from directory: ($dir)"
            (ls $dir | where type == "Dir" or type == "symlink" | get name)
        } else { [] }

        # 合并所有 case，去重
        [$c1, $c2, $c3] | flatten | uniq
    }
    
    process_cases $cases $config $output_root $extra $apply
}