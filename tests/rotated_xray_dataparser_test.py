from pathlib import Path
from unittest.mock import patch

import unittest
import numpy as np

from internal.dataparsers.rotated_xray_dataparser import RotatedXRay
from internal.dataparsers.rotated_xray_dataparser import _init_point_cloud_from_fbp_with_process_mode
from internal.dataset import Dataset

class RotatedXRayDataparserTestCase(unittest.TestCase):
    @property
    def parser(self):
        return RotatedXRay(
            init_point_cloud_mode="FBP",
            init_point_cloud_fbp_phase_max=0.2
        ).instantiate(
            path="data/asoca-diseased__Diseased_17__LCA",
            output_path="/media/data3/sj/Code/gaussian-splatting-lightning/outputs/temp",
            global_rank=0
        )

    def test_rotated_xray_dataparser(self):
        outputs = self.parser.get_outputs()
        image_name, image_path, _, cameras, _ = next(iter(outputs.train_set))
        print(f"{len(outputs.train_set) = }")
        print(f"{image_name = }, {image_path = }")
        print(f"{cameras.camera_center = }")
        print(f"{cameras.world_to_camera = }")
        print(f"{cameras.fov_x = }")
        print(f"{cameras.fov_y = }")
    
    def test_dataset(self):
        outputs = self.parser.get_outputs()
        dataset = Dataset(image_set=outputs.train_set)
        
        batch = next(iter(dataset))
        
        camera, image_info, depth_map = batch
        image_name, gt_image, masked_pixels = image_info
        
        print(f"{len(dataset) =}")
        print(f"{gt_image.shape = }")       # still read as rgb, shape = (C, H, W)
        print(f"{depth_map.shape = }")      # shape = (H, W)
        print(f"{image_name = }")
        print(f"{masked_pixels.shape = }")  # shape = gt_image.shape, dtype = bool
        print(f"{camera.camera_center = }")
        
        #assert
        assert len(gt_image.shape) == 3 and gt_image.shape[0] == 3
        assert len(depth_map.shape) == 2
        assert masked_pixels.shape == gt_image.shape

    def test_fbp_routing_uses_subprocess_by_default(self):
        dummy_xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        with patch(
            "internal.dataparsers.rotated_xray_dataparser._init_point_cloud_from_fbp",
            return_value=np.zeros_like(dummy_xyz),
        ) as direct_fbp, patch(
            "internal.dataparsers.rotated_xray_dataparser._init_point_cloud_from_fbp_in_subprocess",
            return_value=dummy_xyz,
        ) as subprocess_fbp:
            xyz = _init_point_cloud_from_fbp_with_process_mode(
                json_data={},
                image_paths=[],
                frame_indices=[],
                alphas=[],
                phases=[],
                volume_shape=(1, 1, 1),
                volume_affine=np.eye(4),
                num_points=1,
                seed=0,
                use_filter=True,
                phase_max=1.0,
                use_separate_process=True,
            )

        subprocess_fbp.assert_called_once()
        direct_fbp.assert_not_called()
        np.testing.assert_array_equal(xyz, dummy_xyz)

    def test_fbp_routing_can_use_original_flow(self):
        dummy_xyz = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)

        with patch(
            "internal.dataparsers.rotated_xray_dataparser._init_point_cloud_from_fbp",
            return_value=dummy_xyz,
        ) as direct_fbp, patch(
            "internal.dataparsers.rotated_xray_dataparser._init_point_cloud_from_fbp_in_subprocess",
            return_value=np.zeros_like(dummy_xyz),
        ) as subprocess_fbp:
            xyz = _init_point_cloud_from_fbp_with_process_mode(
                json_data={},
                image_paths=[],
                frame_indices=[],
                alphas=[],
                phases=[],
                volume_shape=(1, 1, 1),
                volume_affine=np.eye(4),
                num_points=1,
                seed=0,
                use_filter=True,
                phase_max=1.0,
                use_separate_process=False,
            )

        direct_fbp.assert_called_once()
        subprocess_fbp.assert_not_called()
        np.testing.assert_array_equal(xyz, dummy_xyz)
    
    def time_output(self):
        outputs = self.parser.get_outputs()
        dataset = Dataset(image_set=outputs.train_set)
        for camera, _, _ in dataset:
            phase = camera.time.item()
            print(f"{phase = :.2f}", end='\t')
        
    
if __name__ == '__main__':
    unittest.main()