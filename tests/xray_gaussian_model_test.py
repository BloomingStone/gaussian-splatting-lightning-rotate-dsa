import unittest
from unittest.mock import MagicMock
import torch

# 导入要测试的类
from internal.models.xray_coronary_gaussian import (
    XrayGassianState, 
    XrayGaussianParameterDict,
    XrayCoronaryGaussian,
    XrayCoronaryGaussianModel,
    _split_key
)
from internal.renderers.deformabel_xray_renderer import (
    RenderRes, DeformNetworkConfig, XYZEncodingConfig, TimeEncodingConfig,
    DeformableRendererOptimizationConfig, CoronaryDeformableXrayRenderer,
)

from internal.dataparsers.rotated_xray_dataparser import (
    RotatedXRayDataParser, RotatedXRay
)
from internal.dataset import Dataset

class TestXrayGaussianParameterDict(unittest.TestCase):
    """测试 XrayGaussianParameterDict 类的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.n_coronary = 100
        self.n_background = 200
        
        # 创建测试用的张量
        self.coronary_means = torch.randn(self.n_coronary, 3)
        self.background_means = torch.randn(self.n_background, 3)
        self.coronary_scales = torch.randn(self.n_coronary, 3)
        self.background_scales = torch.randn(self.n_background, 3)
        
        # 创建合并的张量
        self.combined_means = torch.cat([self.coronary_means, self.background_means], dim=0)
        self.combined_scales = torch.cat([self.coronary_scales, self.background_scales], dim=0)
    
    def test_init(self):
        """测试初始化"""
        # 测试默认初始化
        param_dict = XrayGaussianParameterDict()
        self.assertEqual(param_dict.state, XrayGassianState.WHOLE)
        self.assertIsNone(param_dict._n_coronary_gs)
        self.assertIsNone(param_dict._n_background_gs)
        self.assertEqual(len(param_dict.coronary_gs), 0)
        self.assertEqual(len(param_dict.background_gs), 0)
        
        # 测试带参数初始化
        param_dict = XrayGaussianParameterDict(
            state=XrayGassianState.CORONARY,
            n_coronary_gs=self.n_coronary,
            n_background_gs=self.n_background
        )
        self.assertEqual(param_dict.state, XrayGassianState.CORONARY)
        self.assertEqual(param_dict._n_coronary_gs, self.n_coronary)
        self.assertEqual(param_dict._n_background_gs, self.n_background)
    
    def test_state_property(self):
        """测试状态属性"""
        param_dict = XrayGaussianParameterDict()
        
        # 测试设置状态为枚举值
        param_dict.state = XrayGassianState.CORONARY
        self.assertEqual(param_dict.state, XrayGassianState.CORONARY)
        
        # 测试设置状态为字符串
        param_dict.state = "BACKGROUND"
        self.assertEqual(param_dict.state, XrayGassianState.BACKGROUND)
        
        # 测试设置状态为WHOLE
        param_dict.state = "WHOLE"
        self.assertEqual(param_dict.state, XrayGassianState.WHOLE)
    
    def test_init_n_gaussians(self):
        """测试初始化高斯数量"""
        param_dict = XrayGaussianParameterDict()
        
        # 正常情况
        param_dict.init_n_gaussians(self.n_coronary, self.n_background)
        self.assertEqual(param_dict._n_coronary_gs, self.n_coronary)
        self.assertEqual(param_dict._n_background_gs, self.n_background)
        
        # 重复设置应该抛出异常
        with self.assertRaises(AssertionError):
            param_dict.init_n_gaussians(self.n_coronary, self.n_background)
    
    def test_setitem_with_tuple(self):
        """测试使用元组设置参数"""
        param_dict = XrayGaussianParameterDict()
        
        # 使用元组设置参数
        param_dict["means"] = (self.coronary_means, self.background_means)
        
        # 验证设置是否成功
        self.assertEqual(param_dict._n_coronary_gs, self.n_coronary)
        self.assertEqual(param_dict._n_background_gs, self.n_background)
        
        # 验证值是否正确
        means = param_dict["means"]
        coronary_means = means[:param_dict._n_coronary_gs]
        background_means = means[param_dict._n_coronary_gs:]
        self.assertTrue(torch.equal(coronary_means, self.coronary_means))
        self.assertTrue(torch.equal(background_means, self.background_means))
    
    def test_setitem_with_tensor(self):
        """测试使用张量设置参数"""
        param_dict = XrayGaussianParameterDict()
        
        # 先初始化高斯数量
        param_dict.init_n_gaussians(self.n_coronary, self.n_background)
        
        # 使用合并的张量设置参数
        param_dict["means"] = self.combined_means
        
        # 验证值是否正确分割
        means = param_dict["means"]
        coronary_means = means[:param_dict._n_coronary_gs]
        background_means = means[param_dict._n_coronary_gs:]
        self.assertTrue(torch.equal(coronary_means, self.coronary_means))
        self.assertTrue(torch.equal(background_means, self.background_means))
        
        # 测试长度不匹配的情况
        wrong_tensor = torch.randn(self.n_coronary + self.n_background + 10, 3)
        with self.assertRaises(AssertionError):
            param_dict["scales"] = wrong_tensor
    
    def test_setitem_without_initialization(self):
        """测试未初始化高斯数量时使用张量设置参数"""
        param_dict = XrayGaussianParameterDict()
        
        # 未初始化高斯数量时使用张量应该抛出异常
        with self.assertRaises(AssertionError):
            param_dict["means"] = self.combined_means
    
    def test_setitem_with_state(self):
        """测试在不同状态下设置参数"""
        param_dict = XrayGaussianParameterDict()
        
        # 先使用元组初始化
        param_dict["means"] = (self.coronary_means, self.background_means)
        
        # 切换到CORONARY状态
        param_dict.state = XrayGassianState.CORONARY
        
        # 在CORONARY状态下设置参数
        new_coronary_means = torch.randn(self.n_coronary, 3)
        param_dict["means"] = new_coronary_means
        
        # 验证只有冠状动脉参数被更新
        means = param_dict["means_whole"]
        coronary_means = means[:param_dict._n_coronary_gs]
        background_means = means[param_dict._n_coronary_gs:]
        self.assertTrue(torch.equal(coronary_means, new_coronary_means))
        self.assertTrue(torch.equal(background_means, self.background_means))
        
        # 尝试设置新参数应该失败
        with self.assertRaises(AssertionError):
            param_dict["new_param"] = torch.randn(self.n_coronary, 3)
    
    def test_getitem(self):
        """测试获取参数"""
        param_dict = XrayGaussianParameterDict()
        
        # 设置参数
        param_dict["means"] = (self.coronary_means, self.background_means)
        
        # 在WHOLE状态下获取参数
        means = param_dict["means"]
        coronary_means = means[:param_dict._n_coronary_gs]
        background_means = means[param_dict._n_coronary_gs:]
        self.assertTrue(torch.equal(coronary_means, self.coronary_means))
        self.assertTrue(torch.equal(background_means, self.background_means))
        
        # 切换到CORONARY状态
        param_dict.state = XrayGassianState.CORONARY
        coronary_means = param_dict["means"]
        self.assertTrue(torch.equal(coronary_means, self.coronary_means))
        
        # 切换到BACKGROUND状态
        param_dict.state = XrayGassianState.BACKGROUND
        background_means = param_dict["means"]
        self.assertTrue(torch.equal(background_means, self.background_means))
        
        # 测试使用带state的字符串获取参数
        for global_state in XrayGassianState:
            for local_state, means in {
                XrayGassianState.WHOLE: self.combined_means,
                XrayGassianState.CORONARY: self.coronary_means,
                XrayGassianState.BACKGROUND: self.background_means
            }.items():
                param_name = f"means_{local_state.name}"
                self.assertTrue(torch.equal(param_dict[param_name], means))
        
        # 测试使用不存在的参数应该抛出异常
        with self.assertRaises(Exception):
            param_dict["new_param"]
        
        with self.assertRaises(Exception):
            param_dict["new_param_coronary"]
        
        with self.assertRaises(Exception):
            param_dict["new_param_background"]
            
        with self.assertRaises(Exception):
            param_dict["new_param_whole"]
    
    def test_delitem(self):
        """测试删除参数"""
        param_dict = XrayGaussianParameterDict()
        
        # 设置参数
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict["scales"] = (self.coronary_scales, self.background_scales)
        
        # 在WHOLE状态下删除参数
        del param_dict["means"]
        self.assertNotIn("means", param_dict)
        self.assertNotIn("means", param_dict.coronary_gs)
        self.assertNotIn("means", param_dict.background_gs)
        
        # 切换到CORONARY状态
        param_dict.state = XrayGassianState.CORONARY
        
        # 在非WHOLE状态下删除参数应该失败
        with self.assertRaises(Exception):
            del param_dict["scales"]
    
    def test_len(self):
        """测试长度"""
        param_dict = XrayGaussianParameterDict()
        
        # 初始长度为0
        self.assertEqual(len(param_dict), 0)
        
        # 添加参数
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict["scales"] = (self.coronary_scales, self.background_scales)
        
        # 长度应该是2
        self.assertEqual(len(param_dict), 2)
    
    def test_iter(self):
        """测试迭代"""
        param_dict = XrayGaussianParameterDict()
        
        # 添加参数
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict["scales"] = (self.coronary_scales, self.background_scales)
        
        # 测试迭代
        keys = list(param_dict)
        self.assertEqual(keys, ["means", "scales"])
        
        # 测试反向迭代
        reversed_keys = list(reversed(param_dict))
        self.assertEqual(reversed_keys, ["scales", "means"])
    
    def test_contains(self):
        """测试包含检查"""
        param_dict = XrayGaussianParameterDict()
        
        # 添加参数
        param_dict["means"] = (self.coronary_means, self.background_means)
        
        # 测试包含
        self.assertIn("means", param_dict)
        self.assertNotIn("scales", param_dict)
    
    def test_copy(self):
        """测试复制"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict.state = XrayGassianState.CORONARY
        
        # 复制
        copied_dict = param_dict.copy()
        
        # 验证复制结果
        self.assertEqual(copied_dict.state, param_dict.state)
        self.assertEqual(copied_dict._n_coronary_gs, param_dict._n_coronary_gs)
        self.assertEqual(copied_dict._n_background_gs, param_dict._n_background_gs)
        
        # 验证参数值
        copied_dict.state = XrayGassianState.CORONARY
        self.assertTrue(torch.equal(copied_dict["means"], self.coronary_means))
        
        copied_dict.state = XrayGassianState.BACKGROUND
        self.assertTrue(torch.equal(copied_dict["means"], self.background_means))
        
        copied_dict.state = XrayGassianState.WHOLE
        self.assertTrue(torch.equal(copied_dict["means"], self.combined_means))
        
        # 验证是深拷贝
        copied_dict.state = XrayGassianState.WHOLE
        copied_dict["scales"] = (self.coronary_scales, self.background_scales)
        self.assertNotIn("scales", param_dict)
    
    def test_clear(self):
        """测试清空"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict.state = XrayGassianState.CORONARY
        
        # 清空
        param_dict.clear()
        
        # 验证清空结果
        self.assertEqual(len(param_dict), 0)
        self.assertEqual(param_dict.state, XrayGassianState.WHOLE)
    
    def test_popitem(self):
        """测试弹出项"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        
        # 弹出项
        key, value = param_dict.popitem()
        
        # 验证弹出结果
        self.assertEqual(key, "means")
        self.assertTrue(torch.equal(value, self.combined_means))
        self.assertNotIn("means", param_dict)
        
        # 在非WHOLE状态下弹出应该失败
        param_dict.state = XrayGassianState.CORONARY
        with self.assertRaises(AssertionError):
            param_dict["scales"] = (self.coronary_scales, self.background_scales)

        with self.assertRaises(AssertionError):
            param_dict.popitem()

        
    
    def test_keys(self):
        """测试键"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict["scales"] = (self.coronary_scales, self.background_scales)
        
        # 测试键
        keys = list(param_dict.keys())
        self.assertEqual(keys, ["means", "scales"])
    
    def test_items(self):
        """测试项"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict["scales"] = (self.coronary_scales, self.background_scales)
        
        # 在WHOLE状态下测试项
        items = list(param_dict.items())
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][0], "means")
        self.assertTrue(torch.equal(items[0][1], self.combined_means))
        
        # 在CORONARY状态下测试项
        param_dict.state = XrayGassianState.CORONARY
        items = list(param_dict.items())
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][0], "means")
        self.assertTrue(torch.equal(items[0][1], self.coronary_means))
        
        # 在BACKGROUND状态下测试项
        param_dict.state = XrayGassianState.BACKGROUND
        items = list(param_dict.items())
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][0], "means")
        self.assertTrue(torch.equal(items[0][1], self.background_means))
    
    def test_values(self):
        """测试值"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        param_dict["scales"] = (self.coronary_scales, self.background_scales)
        
        # 在WHOLE状态下测试值
        values = list(param_dict.values())
        self.assertEqual(len(values), 2)
        self.assertTrue(torch.equal(values[0], self.combined_means))
        
        # 在CORONARY状态下测试值
        param_dict.state = XrayGassianState.CORONARY
        values = list(param_dict.values())
        self.assertEqual(len(values), 2)
        self.assertTrue(torch.equal(values[0], self.coronary_means))
        
        # 在BACKGROUND状态下测试值
        param_dict.state = XrayGassianState.BACKGROUND
        values = list(param_dict.values())
        self.assertEqual(len(values), 2)
        self.assertTrue(torch.equal(values[0], self.background_means))
    
    def test_update(self):
        """测试更新"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        
        # 使用字典更新
        new_scales = (torch.randn(self.n_coronary, 3), torch.randn(self.n_background, 3))
        param_dict.update({"scales": new_scales})
        
        # 验证更新结果
        self.assertIn("scales", param_dict)
        scales = param_dict["scales"]
        coronary_scales = scales[:param_dict._n_coronary_gs]
        background_scales = scales[param_dict._n_coronary_gs:]
        self.assertTrue(torch.equal(coronary_scales, new_scales[0]))
        self.assertTrue(torch.equal(background_scales, new_scales[1]))
        
        # 使用另一个XrayGaussianParameterDict更新
        other_dict = XrayGaussianParameterDict()
        other_dict.init_n_gaussians(self.n_coronary, self.n_background)
        other_dict["rotations"] = (torch.randn(self.n_coronary, 4), torch.randn(self.n_background, 4))
        
        param_dict.update(other_dict)
        
        # 验证更新结果
        self.assertIn("rotations", param_dict)
        
        # 在非WHOLE状态下更新
        param_dict.state = XrayGassianState.CORONARY
        new_coronary_means = torch.randn(self.n_coronary, 3)
        param_dict.update({"means": new_coronary_means})
        
        # 验证只有冠状动脉参数被更新
        self.assertTrue(torch.equal(param_dict["means_coronary"], new_coronary_means))
        self.assertTrue(torch.equal(param_dict["means_background"], self.background_means))
        
        # 尝试添加新参数应该失败
        with self.assertRaises(AssertionError):
            param_dict.update({"new_param": torch.randn(self.n_coronary, 3)})
    
    def test_extra_repr(self):
        """测试额外表示"""
        param_dict = XrayGaussianParameterDict()
        param_dict["means"] = (self.coronary_means, self.background_means)
        
        # 测试额外表示
        repr_str = param_dict.extra_repr()
        self.assertIn("state = whole", repr_str)
        
        # 在不同状态下测试
        param_dict.state = XrayGassianState.CORONARY
        repr_str = param_dict.extra_repr()
        self.assertIn("state = coronary", repr_str)


class TestSplitKey(unittest.TestCase):
    """测试 _split_key 函数"""
    
    def test_split_key(self):
        """测试键分割"""
        # 测试无后缀的键
        key, state = _split_key("means")
        self.assertEqual(key, "means")
        self.assertIsNone(state)
        
        # 测试有后缀的键
        key, state = _split_key("means_coronary")
        self.assertEqual(key, "means")
        self.assertEqual(state, XrayGassianState.CORONARY)
        
        key, state = _split_key("means_background")
        self.assertEqual(key, "means")
        self.assertEqual(state, XrayGassianState.BACKGROUND)
        
        key, state = _split_key("means_whole")
        self.assertEqual(key, "means")
        self.assertEqual(state, XrayGassianState.WHOLE)
        
        # 测试大小写不敏感
        key, state = _split_key("means_CORONARY")
        self.assertEqual(key, "means")
        self.assertEqual(state, XrayGassianState.CORONARY)


class TestDeformableXrayRender(unittest.TestCase):
    def setUp(self):
        parser = RotatedXRay(
            init_point_cloud_mode="central-line"
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
            deform_network=DeformNetworkConfig(rotate_xyz=False),
            xyz_encoding=XYZEncodingConfig(),
            time_encoding=TimeEncodingConfig(),
            optimization=DeformableRendererOptimizationConfig(),
            exp_neg_img=True,
        )
        self.render.setup("fit", self.lightning_module)
        self.render.to('cuda')
        self.render.training_setup(self.lightning_module)
    
    def test_forward(self):
        camera, image_info, depth_map = self.batch
        print(camera.camera_center)
        image_name, gt_image, masked_pixels = image_info
        self.render.eval()
        res = self.render.forward(
            step=0,
            module=self.lightning_module,
            viewpoint_camera=camera.to_device(self.device),
            pc=self.gs_model.to(self.device),
        )
        
        for key, value in self.gs_model.get_properties().items():
            assert value.requires_grad
        
        C, H, W = gt_image.shape
        assert res.gray_image_coronary.shape == (1, H, W)
        assert res.gray_image_whole.shape == (1, H, W)
        
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
        
        output_image(res.gray_image_coronary, "gray_image_coronary")
        output_image(res.gray_image_whole, "gray_image_whole")

if __name__ == "__main__":
    # unittest.main()
    test = TestDeformableXrayRender()
    test.setUp()
    test.test_forward()
