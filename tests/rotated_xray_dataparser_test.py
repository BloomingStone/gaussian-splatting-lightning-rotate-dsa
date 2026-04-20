from pathlib import Path

import unittest
from internal.dataparsers.rotated_xray_dataparser import RotatedXRay 
from internal.datasets.vanilla_dataset import Dataset

class RotatedXRayDataparserTestCase(unittest.TestCase):
    @property
    def parser(self):
        return RotatedXRay(
            init_point_cloud_mode="central-line"
        ).instantiate(
            path="data/volume_dvf_reader_multipli_contrast_LCA",
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
    
    def time_output(self):
        outputs = self.parser.get_outputs()
        dataset = Dataset(image_set=outputs.train_set)
        for camera, _, _ in dataset:
            phase = camera.time.item()
            print(f"{phase = :.2f}", end='\t')
        
    
if __name__ == '__main__':
    unittest.main()