import json
import os.path
from typing import Literal
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule, Trainer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from .dataparsers.dataparser import DataParser, DataParserBuilder, collate_fn



Stage = Literal["train", "val", "test"]

class DataModule(LightningDataModule):
    def __init__(
            self,
            path: str,
            parser: DataParserBuilder,
            val_on_train: bool = False,
            num_workers: int|dict[Stage, int] = 2,
    ) -> None:
        r"""Load dataset

            Args:
                path: the path to the dataset

                type: the dataset type
        """

        super().__init__()
        
        self.parser = parser

        self.save_hyperparameters()
        
        self.path = path
        self.val_on_train =val_on_train
        
        def _to_dict(v):
            if not isinstance(v, dict):
                return {
                    "train": v,
                    "val": v,
                    "test": v,
                }
            else:
                assert set(v.keys()) == {"train", "val", "test"}
                return v
        
        self.num_workers = _to_dict(num_workers)


    def setup(self, stage: str) -> None:
        super().setup(stage)
        output_path = self.get_trainer().lightning_module.hparams["output_path"]

        # store global rank, will be used as the seed of the CacheDataLoader
        self.global_rank = self.get_trainer().global_rank

        self.dataparser = self.parser.build()

        # load dataset
        self.dataparser_outputs = self.dataparser.get_outputs(Path(self.path))

        self.prune_extent = self.dataparser_outputs.camera_extent

        # convert point cloud
        self.point_cloud = self.dataparser_outputs.point_cloud

        # write some files that SIBR_viewer required
        if self.global_rank == 0 and stage == "fit":
            # write cameras.json
            camera_to_world = torch.linalg.inv(
                torch.transpose(self.dataparser_outputs.train_set.cameras.world_to_camera, 1, 2)
            ).cpu().numpy()
            cameras = []
            for idx, image in enumerate(iter(self.dataparser_outputs.train_set)):
                camera, image_info, _ = image
                image_name, _, _ = image_info
                cameras.append({
                    'id': idx,
                    'img_name': image_name,
                    'width': int(camera.width),
                    'height': int(camera.height),
                    'position': camera_to_world[idx, :3, 3].tolist(),
                    'rotation': [x.tolist() for x in camera_to_world[idx, :3, :3]],
                    'fy': float(camera.fy),
                    'fx': float(camera.fx),
                    'cx': camera.cx.item(),
                    'cy': camera.cy.item(),
                    'time': camera.time.item() if camera.time is not None else None
                })
            with open(os.path.join(output_path, "cameras.json"), "w") as f:
                json.dump(cameras, f, indent=4, ensure_ascii=False)

            # save input point cloud to vtp file
            pd = pv.PolyData(self.dataparser_outputs.point_cloud.xyz.astype(np.float32))
            pd.point_data["rgb"] = self.dataparser_outputs.point_cloud.rgb.astype(np.float32)
            pd.save(os.path.join(output_path, "input.vtp"))

    
    def get_trainer(self) -> Trainer:
        assert self.trainer is not None, "trainer is not set yet"
        return self.trainer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        n_workers = self.num_workers["train"]
        return DataLoader(
            self.dataparser_outputs.train_set,
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_fn,
            persistent_workers=(n_workers > 0),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_on_train is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.test_set
        
        n_workers = self.num_workers["test"]
        return DataLoader(
            image_set,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_fn,
            persistent_workers=(n_workers > 0),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_on_train is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.val_set
        
        n_workers = self.num_workers["val"]
        return DataLoader(
            image_set,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_fn,
            persistent_workers=(n_workers > 0),
        )
