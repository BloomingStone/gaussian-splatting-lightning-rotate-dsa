import unittest
from unittest.mock import MagicMock
import torch

# 导入要测试的类
from internal.models.xray_coronary_gaussian import (
    XrayCoronaryGaussian,
    XrayCoronaryGaussianModel,
)
from internal.renderers.deformabel_xray_renderer import (
    RenderRes, DeformNetworkConfig, SegNetworkConfig, XYZEncodingConfig, TimeEncodingConfig,
    DeformableRendererOptimizationConfig, CoronaryDeformableXrayRenderer,
)

from internal.dataparsers.rotated_xray_dataparser import (
    RotatedXRayDataParser, RotatedXRay
)
from internal.dataset import Dataset

class TestDeformableXrayRender(unittest.TestCase):
    def setUp(self):
        parser = RotatedXRay(
            init_point_cloud_mode="label"
        ).instantiate(
            path="/media/data3/sj/Code/Gen4D/test/output/intergration_full/volume_dvf_reader_multipli_contrast_LCA",
            output_path="/media/data3/sj/Code/gaussian-splatting-lightning/outputs/temp",
            global_rank=0
        )
        
        outputs = parser.get_outputs()
        
        self.device = torch.device("cuda:0")
        dataset = Dataset(image_set=outputs.train_set, camera_device=self.device, image_device=self.device)
        
        self.batch = next(iter(dataset))
        
        self.lightning_module = MagicMock()
        self.lightning_module.trainer.datamodule.dataparser_outputs.train_set = outputs.train_set
        self.lightning_module.trainer.datamodule.dataparser_outputs.camera_extent = outputs.camera_extent
        self.lightning_module.on_after_backward_hooks = []
        
        self.gs_model = XrayCoronaryGaussian().instantiate()
        self.gs_model.setup_from_pcd(xyz=outputs.point_cloud.xyz, rgb=None)
        self.gs_model.training_setup(self.lightning_module)
        self.gs_model = self.gs_model.to(self.device)
        
        self.render = CoronaryDeformableXrayRenderer(
            deform_network=DeformNetworkConfig(),
            segmentation_network=SegNetworkConfig(),
            xyz_encoding=XYZEncodingConfig(),
            time_encoding=TimeEncodingConfig(),
            optimization=DeformableRendererOptimizationConfig(),
        )
        self.render.setup("fit", self.lightning_module)
        self.render.to('cuda')
        self.render.training_setup(self.lightning_module)
    
    def test_training_forward(self):
        camera, image_info, depth_map = self.batch
        print(camera.camera_center)
        image_name, gt_image, masked_pixels = image_info
        res = self.render.forward(
            step=0,
            module=self.lightning_module,
            viewpoint_camera=camera.to_device(self.device),
            pc=self.gs_model.to(self.device),
        )
        
        for key, value in self.gs_model.get_properties().items():
            assert value.requires_grad
        
        C, H, W = gt_image.shape
        assert res.gray_image.shape == (1, H, W)
        assert res.depth.shape == (1, H, W)
        assert res.coronary_probs.shape == (1, H, W)
        
        from pathlib import Path
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        import numpy as np
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        def output_image(image: torch.Tensor, name: str):
            image_np = image.detach().cpu().numpy()
            if image_np.max() > 0:
                vmin = np.min(image_np[image_np>0])
            else:
                vmin = np.min(image_np)
            plt.imshow(image_np.squeeze(), vmin=vmin, cmap="gray")
            plt.savefig(output_dir / f"{name}.png")
            plt.close()
        
        output_image(res.gray_image, "gray_image_whole")
        output_image(res.depth, "depth")
        output_image(res.coronary_probs, "coronary_probs")

if __name__ == "__main__":
    # unittest.main()
    test = TestDeformableXrayRender()
    test.setUp()
    test.test_training_forward()
