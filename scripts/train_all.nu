#!/usr/bin/env nu

def --wrapped main [
    data_root: path,        # 数据目录
    config: path,           # 配置文件
    output_root: path,      # 输出目录
    --dryrun                # 仅打印命令，不执行
    ...extra: string        # 额外参数（列表），直接透传给 gs-fit
                            # 如: --logger wandb --trainer.max_steps 5000
] {
    let cases = (
        ls $data_root
        | where type == dir or type == symlink
    )

    let total = ($cases | length)

    print $"Found ($total) cases in ($data_root)."
    print ($cases | first 5)

    for item in ($cases | enumerate) {
        let idx = ($item.index + 1)
        let d = $item.item

        let exp_name = (
            $"(($d.name | path dirname | path basename))_(($config | path parse | get stem))"
        )
        let case_name = ($d.name | path basename)

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
            "--data.path", $d.name,
            "-n", $exp_name,
            "-v", $case_name,
            ...$extra
        ]

        if $dryrun {
            print $"pixi run gs-fit -- ($all_args | str join ' ')"
            continue
        }

        let cmd = $"pixi run gs-fit -- ($all_args | str join ' ')"

        try {
            ^pixi ...[run gs-fit --] ++ $all_args
        } catch {|e|
            let log_line = [
                $"[(date now | format date '%Y-%m-%d %H:%M:%S')] FAILED: ($case_name)"
                $"    config: ($config)"
                $"    data:   ($d.name)"
                $"    output: ($output_case_dir)"
                $"    cmd:    ($cmd)"
                ""
            ] | str join "\n"

            $log_line | save --append ./error.log
            print $"[($idx)/($total)] FAILED ($case_name) (see error.log)"
        }
    }
}