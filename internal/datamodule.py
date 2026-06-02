import json
import os.path
from typing import Literal
from pathlib import Path
import math
from queue import Queue
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyvista as pv
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from lightning import LightningDataModule, Trainer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from .dataparsers.dataparser import DataParser, DataParserBuilder, collate_fn, GSDataset


Stage = Literal["train", "val", "test"]

class CacheDataLoader(DataLoader):
    def __init__(
            self,
            dataset: GSDataset,
            max_cache_num: int,
            shuffle: bool,
            seed: int = -1,
            distributed: bool = False,
            world_size: int = -1,
            global_rank: int = -1,
            async_caching: bool = False,
            **kwargs,
    ):
        assert kwargs.get("batch_size", 1) == 1, "only batch_size=1 is supported"

        self.dataset = dataset

        super().__init__(dataset=dataset, **kwargs)

        self.shuffle = shuffle
        self.max_cache_num = max_cache_num

        # image indices to use
        self.indices = list(range(len(self.dataset)))
        if distributed is True and self.max_cache_num != 0:
            assert world_size > 0
            assert global_rank >= 0
            image_num_to_use = math.ceil(len(self.indices) / world_size)
            start = global_rank * image_num_to_use
            end = start + image_num_to_use
            indices = self.indices[start:end]
            indices += self.indices[:image_num_to_use - len(indices)]
            self.indices = indices

            print("#{} distributed indices (total: {}): {}".format(os.getpid(), len(self.indices), self.indices))

        # cache all images if max_cache_num > len(dataset)
        if self.max_cache_num >= len(self.indices):
            self.max_cache_num = -1

        self.num_workers = kwargs.get("num_workers", 0)

        if self.max_cache_num < 0:
            # cache all data
            print("cache all images")
            self.cached = self._cache_data(self.indices)

        # use dedicated random number generator foreach dataloader
        if self.shuffle is True:
            assert seed >= 0, "seed must be provided when shuffle=True"
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
            print("#{} dataloader seed to {}".format(os.getpid(), seed))

        self.async_caching = async_caching and self.max_cache_num > 0
        self.cache_output_queue = None
        self.cache_thread = None
        self.stop_caching = False
        if self.async_caching:
            self.cache_output_queue = Queue(maxsize=1)
            self.cache_thread = threading.Thread(target=self._async_cache)
            self.cache_thread.start()

    def _async_cache(self):
        # TODO: GC will freeze program a while
        while not self.stop_caching:
            if self.shuffle is True:
                indices = torch.randperm(len(self.indices), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            else:
                indices = self.indices.copy()

            not_cached = indices.copy()

            while not_cached and not self.stop_caching:
                # select self.max_cache_num images
                to_cache = not_cached[:self.max_cache_num]
                del not_cached[:self.max_cache_num]

                assert self.cache_output_queue is not None
                self.cache_output_queue.put(None)  # simulate a queue with zero size
                self.cache_output_queue.put(self._cache_data(to_cache, pbar_leave=False))

    def _cache_data(self, indices: list, pbar_leave: bool = True):
        cached = []
        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                for i in tqdm(
                        e.map(self.dataset.__getitem__, indices),
                        total=len(indices),
                        desc="#{} caching images (1st: {})".format(os.getpid(), indices[0]),
                        leave=pbar_leave,
                ):
                    cached.append(i)
        else:
            for i in tqdm(indices, desc="#{} loading images (1st: {})".format(os.getpid(), indices[0]), leave=pbar_leave):
                cached.append(self.dataset.__getitem__(i))

        return cached

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __iter__(self):
        # TODO: support batching
        if self.max_cache_num < 0:
            if self.shuffle is True:
                indices = torch.randperm(len(self.cached), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            else:
                indices = list(range(len(self.cached)))

            for i in indices:
                yield self.cached[i]
        else:
            if self.shuffle is True:
                indices = torch.randperm(len(self.indices), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            else:
                indices = self.indices.copy()

            # print("#{} self.max_cache_num={}, indices: {}".format(os.getpid(), self.max_cache_num, indices))

            if self.max_cache_num == 0:
                # no cache
                for i in indices:
                    yield self.__getitem__(i)
            else:
                # cache
                # the list contains the data have not been cached
                not_cached = indices.copy()

                if self.async_caching:
                    assert self.cache_output_queue is not None
                    while True:
                        cached = self.cache_output_queue.get()  # setting to None allows GC
                        assert cached is None
                        cached = self.cache_output_queue.get()
                        for i in cached:
                            yield i
                else:
                    while not_cached:
                        # select self.max_cache_num images
                        to_cache = not_cached[:self.max_cache_num]
                        del not_cached[:self.max_cache_num]

                        # cache
                        try:
                            del cached
                        except:
                            pass
                        cached = self._cache_data(to_cache, pbar_leave=False)

                        for i in cached:
                            yield i




class DataModule(LightningDataModule):
    def __init__(
            self,
            path: str,
            parser: DataParserBuilder,
            val_on_train: bool = False,
            num_workers: int|dict[Stage, int] = 2,
            max_cache_num=-1,
            async_caching=False,
            shuffle_seed: int = 42,
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
        self.max_cache_num = max_cache_num
        self.async_caching = async_caching
        self.shuffle_seed = shuffle_seed


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
            cameras = self.dataparser_outputs.train_set.cameras
            image_names = self.dataparser_outputs.train_set.image_names
            sibr_cameras = []
            for idx, camera, image_name in zip(range(len(cameras)), cameras, image_names):
                sibr_cameras.append({
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
                    'time': camera.time.item()
                })
            with open(os.path.join(output_path, "cameras.json"), "w") as f:
                json.dump(sibr_cameras, f, indent=4, ensure_ascii=False)

            # save input point cloud to vtp file
            pd = pv.PolyData(self.dataparser_outputs.point_cloud.xyz.astype(np.float32))
            pd.point_data["rgb"] = self.dataparser_outputs.point_cloud.rgb.astype(np.float32)
            pd.save(os.path.join(output_path, "input.vtp"))

    
    def get_trainer(self) -> Trainer:
        assert self.trainer is not None, "trainer is not set yet"
        return self.trainer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        n_workers = self.num_workers["train"]
        return CacheDataLoader(
            self.dataparser_outputs.train_set,
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_fn,
            max_cache_num=self.max_cache_num,
            async_caching=self.async_caching,
            seed=self.shuffle_seed,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_on_train is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.test_set
        
        n_workers = self.num_workers["test"]
        return CacheDataLoader(
            image_set,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_fn,
            max_cache_num=self.max_cache_num,
            async_caching=self.async_caching,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_on_train is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.val_set
        
        n_workers = self.num_workers["val"]
        return CacheDataLoader(
            image_set,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_fn,
            max_cache_num=self.max_cache_num,
            async_caching=self.async_caching,
        )
