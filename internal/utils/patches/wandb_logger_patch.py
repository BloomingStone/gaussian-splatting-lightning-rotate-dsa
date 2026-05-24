from typing import Union, Any, Dict
from argparse import Namespace


try:
    from lightning.fabric.utilities.rank_zero import rank_zero_only
    from lightning.fabric.utilities.logger import _convert_params, _sanitize_callable_params
    from lightning.pytorch.loggers.wandb import WandbLogger

    vanilla_log_hyperparams = WandbLogger.log_hyperparams

    @rank_zero_only
    def patched_log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        try:
            params = _convert_params(params)
            params = _sanitize_callable_params(params)
        except:
            return vanilla_log_hyperparams(self, params)
        self.experiment.config.update(params, allow_val_change=True)

    WandbLogger.log_hyperparams = patched_log_hyperparams
except Exception as e:
    print("[WARNING]An error occurred on patching WandbLogger: {}".format(e))
