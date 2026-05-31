from pathlib import Path
import torch
import jsonargparse
from jsonargparse import Namespace
from jsonargparse._typehints import subclass_spec_as_namespace
from typing import Optional, Union, List, Literal
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser

from .utils.patches import fix_lightning_save_hyperparameters
from .utils.patches import wandb_logger_patch


def discard_init_args_on_class_path_change(parser_or_action, prev_val, value):
    """
    jsonargparse will reuse args presenting in user specified instance from the default one,
    which means that parameter with same name in different class can not have different default value,
    this function prevent reusing
    """

    if prev_val and "init_args" in prev_val and prev_val["class_path"] != value["class_path"]:
        prev_val = subclass_spec_as_namespace(prev_val)
        # pop all args
        assert prev_val is not None
        for key, val in list(prev_val.init_args.__dict__.items()):
            prev_val.init_args.pop(key)


jsonargparse._typehints.discard_init_args_on_class_path_change = discard_init_args_on_class_path_change     # type: ignore


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--max_steps",
            type=Optional[int], 
            default=None
        )
        parser.add_argument(
            "--max_epochs", 
            type=Optional[int], 
            default=None
        )
        parser.add_argument(
            "--name", "-n", 
            type=Optional[str], 
            default=None,
            help="the training result output path will be 'output/name'"
        )
        parser.add_argument(
            "--version", "-v", 
            type=Optional[str], 
            default=None,
            help="the training result output path will be 'output/name/version'"
        )
        
        # TODO: add max_steps to save_iterations, but need to compatible with --max_steps < 0 & --max_epochs > 0
        parser.add_argument(
            "--save_iterations", 
            type=List[int], 
            default=[7_000, 30_000]
        )
        parser.add_argument(
            "--logger", 
            type=str, 
            default="tensorboard"
        )
        parser.add_argument(
            "--project", 
            type=str, 
            default="Gaussian-Splatting", 
            help="WanDB project name"
        )
        parser.add_argument(
            "--output", 
            type=str, 
            default=Path(__file__).parent.parent / "outputs",
            help="the base directory of the output"
        )
        parser.add_argument(
            "--float32_matmul_precision", "-f", 
            type=Optional[Literal["medium", "high", "highest"]], 
            default=None
        )
        parser.add_argument(
            "--viewer", 
            action="store_true", 
            default=False
        )
        parser.add_argument(
            "--save_val", 
            action="store_true", 
            default=False,
            help="Whether save images rendered during validation/test to files"
        )
        parser.add_argument(
            "--val_train", 
            action="store_true", 
            default=False,
            help="Whether use train set to do validation"
        )
        parser.add_argument(
            "--cache_all_images", 
            action="store_true", 
            default=False,
            help="Speedup validation/test by caching all images. Images in train set is cached by default."
        )
        parser.add_argument(
            "--pbar_rate", 
            type=int, 
            default=None
        )

    def before_instantiate_classes(self) -> None:
        config = getattr(self.config, self.config.subcommand)
        if config.name is None:
            # auto set experiment name base on --data.path
            config.name = "_".join(config.data.path.strip("/").split("/")[-3:])
            print("auto determine experiment name: {}".format(config.name))

        if config.max_steps is not None:
            config.trainer.max_steps = config.max_steps
        if config.max_epochs is not None:
            config.trainer.max_epochs = config.max_epochs

        # build output path
        # - <output> / <name> / [<version>] /
        # |- checkpoints
        # | |- epoch=xxx-step=yyy.ckpt
        # |- val
        # | |- epoch=xxx-step=yyy
        # | | |- zzz.jpg
        # |- point_cloud
        # | |- iteration_zzz.vtp
        # |- volumes
        # | |- volume__epoch=xxx-step=yyy.nii.gz
        # | |- dxyz_volume__epoch=xxx-step=yyy.nii.gz
        # | lightning_logs
        # |- cameras.json
        # |- cfg_args
        # |- input.vtp
        
        output_path = Path(config.output) / config.name
        if config.version is not None:
            output_path = output_path / config.version
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"output path: {output_path}")
        config.model.output_path = str(output_path)

        # ckpt_path from lightning
        if config.ckpt_path == "last":
            config.ckpt_path = _search_checkpoint(output_path)

        if self.config.subcommand == "fit":
            if config.ckpt_path is None:
                assert not (output_path / "point_cloud").exists() and not (output_path / "checkpoints").exists(), (
                    "checkpoint or point cloud output already exists in '{}', \n"
                    "please specific a different experiment name (-n) or version (-v)".format(output_path)
                )
        else:
            # disable logger
            config.logger = "None"
            # disable config saveing
            self.save_config_callback = None
            # find checkpoint automatically if not provided
            if config.ckpt_path is None:
                config.ckpt_path = _search_checkpoint(output_path)

        # build logger
        logger_config = Namespace(
            class_path=None,
            init_args=Namespace(
                save_dir=output_path,
            ),
        )

        if config.logger == "tensorboard":
            logger_config.class_path = "lightning.pytorch.loggers.TensorBoardLogger"
        elif config.logger == "wandb":
            logger_config.class_path = "lightning.pytorch.loggers.WandbLogger"
            wandb_name = config.name
            if config.version is not None:
                wandb_name = "{}_{}".format(wandb_name, config.version)
            setattr(logger_config.init_args, "name", wandb_name)
            setattr(logger_config.init_args, "project", config.project)
        elif config.logger == "none" or config.logger == "None" or config.logger == "false" or config.logger == "False":
            logger_config = False
        else:
            logger_config.class_path = config.logger

        config.trainer.logger = logger_config

        # set torch float32_matmul_precision
        if config.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(config.float32_matmul_precision)

        # set web viewer
        config.model.web_viewer = config.viewer

        if config.save_val is True:
            callbacks = getattr(config.trainer, "callbacks", None)
            if callbacks is not None:
                for callback in callbacks:
                    if _update_callback_init_value(
                        callback=callback,
                        callback_class_path="internal.callbacks.save_image.SaveImage",
                        init_arg_name="save_val_output",
                        init_arg_value=True,
                    ):
                        break

        # route --save_iterations to SceneSaver callback
        if config.save_iterations is not None:
            callbacks = getattr(config.trainer, "callbacks", None)
            if callbacks is not None:
                for callback in callbacks:
                    if _update_callback_init_value(
                        callback=callback,
                        callback_class_path="internal.callbacks.scene_saver.SceneSaver",
                        init_arg_name="save_iterations",
                        init_arg_value=config.save_iterations,
                    ):
                        break

        config.data.val_on_train = config.val_train

        # set refresh rate of the progress bar
        self._set_pbar_rate()
    
    
    def _set_pbar_rate(self) -> None:
        if self.config.pbar_rate is None:
            return
        
        callbacks = getattr(self.trainer, "callbacks", None)
        if callbacks is None:
            return
        
        for callback in callbacks:
            if _update_callback_init_value(
                callback=callback,
                callback_class_path="internal.callbacks.callbacks.ProgressBar",
                init_arg_name="refresh_rate",
                init_arg_value=self.config.pbar_rate,
            ):
                break


def _search_checkpoint(path: Path) -> Path:
    from internal.utils.guassian_utils.gaussian_model_loader import GaussianModelLoader
    
    ckpt_path = GaussianModelLoader.search_load_file(path)
    assert ckpt_path.suffix == ".ckpt", "not a checkpoint can be found in {}".format(path)
    print("Auto select checkpoint file: {}".format(ckpt_path))
    return ckpt_path


def _update_callback_init_value(
    callback: Namespace,
    callback_class_path: str,
    init_arg_name: str,
    init_arg_value: Union[str, int, float, bool],
) -> bool:
    class_path = getattr(callback, "class_path", None)
    if class_path is None:
        return False
    if class_path != callback_class_path:
        return False
    
    init_args = getattr(callback, "init_args", None)
    if init_args is None:
        if isinstance(callback, dict):
            init_args = callback.setdefault("init_args", Namespace())
        else:
            callback.init_args = Namespace()
            init_args = callback.init_args
    
    if isinstance(init_args, dict):
        init_args[init_arg_name] = init_arg_value
        return True
    else:
        setattr(init_args, init_arg_name, init_arg_value)
        return True
