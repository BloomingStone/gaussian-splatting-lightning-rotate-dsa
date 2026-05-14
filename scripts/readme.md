```bash
pixi run python scripts/experiment_manager.py --config-name smoke train.results_root=outputs/experiments_asoca_smoke1 train.dry_run=false train.run_test=true train.skip_existing=false train.case_filter.include_pattern=
'Diseased_04' train.extra_args='["--max_steps","1","--save_iterations","[1]","--save_val"]'
```