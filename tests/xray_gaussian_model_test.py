import unittest
from unittest.mock import MagicMock
import torch
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# 导入要测试的类
from internal.models.xray_coronary_gaussian import (
    XrayCoronaryGaussian,
)
from internal.renderers.deformabel_xray_renderer import (
    DeformableRendererOptimizationConfig, CoronaryDeformableXrayRenderer,
)
from internal.deform_models import HashGridDeformConfig

from internal.dataparsers.rotated_xray_dataparser import RotatedXRay
from internal.dataset import Dataset
from internal.savers.x_ray_saver import XRaySaver

class TestDeformableXrayRenderAndSaver(unittest.TestCase):
    def setUp(self):
        parser = RotatedXRay(
            init_point_cloud_mode="random"
        ).instantiate(
            path="data/volume_dvf_reader_multipli_contrast_LCA",
            output_path="outputs/temp",
            global_rank=0
        )
        
        outputs = parser.get_outputs()
        
        self.device = torch.device("cuda:0")
        dataset = Dataset(image_set=outputs.train_set, camera_device=self.device, image_device=self.device)
        self.batch = next(iter(dataset))

        lightning_module = MagicMock()
        lightning_module.trainer = MagicMock()
        lightning_module.trainer.datamodule.dataparser_outputs.train_set = outputs.train_set
        lightning_module.trainer.datamodule.dataparser_outputs.camera_extent = outputs.camera_extent
        lightning_module.trainer.strategy = None
        lightning_module.trainer.current_epoch = 0
        lightning_module.trainer.global_step = 0
        lightning_module.trainer.global_rank = 0
        lightning_module.trainer.save_checkpoint = lambda path: None
        lightning_module.hparams = {
            "output_path": Path(__file__).parent / "outputs"
        }
        lightning_module.on_after_backward_hooks = []
        
        gs_model = XrayCoronaryGaussian().instantiate()
        gs_model.setup_from_pcd(xyz=outputs.point_cloud.xyz, rgb=None)
        gs_model.training_setup(lightning_module)
        gs_model = gs_model.to(self.device)
        
        render = CoronaryDeformableXrayRenderer(
            optimization_config=DeformableRendererOptimizationConfig(),
            deform_model_config=HashGridDeformConfig(),
        )
        render.setup("fit", lightning_module)
        render.to('cuda')
        render.training_setup(lightning_module)
        
        lightning_module.gaussian_model = gs_model
        lightning_module.renderer = render
        
        self.lightning_module = lightning_module
        self.gs_model = gs_model
        self.renderer = render
        
        self.saver = XRaySaver().instantiate()
    
    def test_training_forward(self):
        camera, image_info, depth_map = self.batch
        print(camera.camera_center)
        image_name, gt_image, masked_pixels = image_info
        res = self.renderer.forward(
            step=0,
            module=self.lightning_module,
            viewpoint_camera=camera.to_device(self.device),
            pc=self.gs_model.to(self.device),
        )
        
        for key, value in self.gs_model.get_properties().items():
            assert value.requires_grad
        
        C, H, W = gt_image.shape
        assert res.gray_coronary is not None
        assert res.gray_coronary.shape == (1, H, W)
        assert res.gray_image.shape == (1, H, W)
        
        matplotlib.use("Agg")
        
        output_dir = self.lightning_module.hparams["output_path"]
        output_dir.mkdir(exist_ok=True)
        def output_image(image: torch.Tensor, name: str):
            image_np = image.detach().cpu().numpy()
            plt.imshow(image_np.squeeze(), cmap="gray")
            plt.savefig(output_dir / f"{name}.png")
            plt.close()
        
        output_image(res.gray_image, "gray_image_whole")

    def test_saver(self):
        self.saver.save(self.lightning_module)
        
if __name__ == "__main__":
    # unittest.main()
    test = TestDeformableXrayRenderAndSaver()
    test.setUp()
    test.test_training_forward()
    test.test_saver()
