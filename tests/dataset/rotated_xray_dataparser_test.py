from pathlib import Path

import unittest
from internal.dataparsers.rotated_xray_dataparser import RotatedXRayDataParser, RotatedXRay

class RotatedXRayDataparserTestCase(unittest.TestCase):
    def test_rotated_xray_dataparser(self):
        input_dir = Path("/media/data3/sj/Code/Gen4D/test/output/intergration/volume_dvf_reader_multipli_contrast_LCA")
        output_dir = str(Path(__file__).parent / "output")
        dataparser = RotatedXRay().instantiate(
            path=str(input_dir),
            output_path=output_dir,
            global_rank=0
        )
        dataparser_outputs = dataparser.get_outputs()
        image_name, image_path, _, cameras, _ = next(iter(dataparser_outputs.train_set))
        print(image_name, image_path)
        print(cameras.camera_center)
        print(cameras.world_to_camera)
        print(cameras.fov_x)
        print(cameras.fov_y)
            
            

if __name__ == '__main__':
    unittest.main()