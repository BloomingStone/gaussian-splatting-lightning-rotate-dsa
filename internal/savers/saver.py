from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor

from lightning import LightningModule

from internal.instantiate_config import Instantiable


@dataclass
class Saver(Instantiable, ABC):
    @abstractmethod
    def instantiate(self, *args, **kwargs) -> "SaverModule":
        pass

class SaverModule(ABC):
    def _save_ckpt(self, pl_module: LightningModule):
        epoch = pl_module.trainer.current_epoch
        step = pl_module.trainer.global_step
        output_root = Path(pl_module.hparams["output_path"])
        ckpt_dir = output_root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch={epoch}-step={step}.ckpt"
        
        pl_module.trainer.save_checkpoint(ckpt_path)
    
    @abstractmethod
    def save(self, pl_module: LightningModule):
        pass


class ThreadedSaverModule(SaverModule):
    """Base class for saver modules that write outputs asynchronously in a background thread."""

    def __init__(self, thread_name_prefix: str = "save"):
        super().__init__()
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._pending_save: Future | None = None

    def _submit_save_task(self, fn, *args, **kwargs):
        """Wait for any pending save to finish, then submit a new task."""
        if self._pending_save is not None and not self._pending_save.done():
            self._pending_save.result()
        self._pending_save = self._save_executor.submit(fn, *args, **kwargs)

    def __del__(self):
        self._save_executor.shutdown(wait=False)