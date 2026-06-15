#!/usr/bin/env nu

def main [
    data_root: path,      # 数据目录
    config: path,         # 配置文件
    output_root: path,    # 输出目录

    --wandb          # 使用 wandb
    --dryrun         # 仅打印命令，不执行
] {
    if ($data_root | path exists) {
        print $"Data root: ($data_root)"
    } else {
        print $"Data root ($data_root) does not exist."
        exit 1
    }

    if ($config | path exists) {
        print $"Config: ($config)"
    } else {
        print $"Config ($config) does not exist."
        exit 1
    }

    if ($output_root | path exists) {
        print $"Output root: ($output_root)"
    } else {
        print $"Output root ($output_root) does not exist. Creating it."
        mkdir $output_root
    }

    let cases = (
        ls $data_root
        | where type == dir or type == symlink
    )

    let total = ($cases | length)

    print $"Found ($total) cases in ($data_root)."
    print ($cases | first 5)

    let exp_name = (
        $"(($data_root | path basename))_(($config | path parse | get stem))"
    )

    for item in ($cases | enumerate) {
        let idx = ($item.index + 1)
        let d = $item.item

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

        if $dryrun {
            print $"python main.py fit --output ($output_root) --config ($config) --data.path ($d.name) -n ($exp_name) -v ($case_name)"
            continue
        }

        if $wandb {
            ^pixi ...[
                run gs-fit
                --
                --output $output_root 
                --config $config 
                --data.path $d.name
                -n $exp_name
                -v $case_name
                --logger wandb
            ]
        } else {
            ^python ...[
                main.py fit
                --output $output_root
                --config $config
                --data.path $d.name
                -n $exp_name
                -v $case_name
            ]
        }
    }
}