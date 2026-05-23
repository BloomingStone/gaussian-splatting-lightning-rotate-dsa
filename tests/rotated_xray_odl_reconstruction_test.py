from __future__ import annotations

from typing import cast
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

from internal.dataparsers.rotated_xray_dataparser import (
    RotatedXRay,
    RotatedXRayDataParser,
    _get_frames_list,
    _preprocess_indices_alphas,
)


class RotatedXRayOdlReconstructionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data_root = Path("data/asoca-diseased__Diseased_17__LCA")
        self.output_dir = Path(__file__).parent / "outputs" / "odl_reconstruction"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        parser = RotatedXRay(
            mode="reconstruction",
            init_point_cloud_mode="FBP",
            init_point_cloud_num=50_000,
            init_point_cloud_fbp_phase_max=0.2,
        ).instantiate(
            path=str(self.data_root),
            output_path=str(self.output_dir),
            global_rank=0,
        )
        
        self.parser = cast(RotatedXRayDataParser, parser)

    def test_reconstruct_and_save_volume_and_point_cloud(self) -> None:
        xyz, volume = self.parser._init_point_cloud_from_fbp()
        
        
        
        volume_slize_path = self.output_dir / "volume_slice.png"
        volume_slice_z = self.parser.volume_shape[2] // 2
        plt.imshow(volume[:, :, volume_slice_z], cmap="gray")
        plt.axis("off")
        plt.savefig(volume_slize_path, bbox_inches="tight", pad_inches=0)

        volume_path = self.output_dir / "volume.nii.gz"
        point_cloud_path = self.output_dir / "point_cloud.npz"
        
        outputs = self.parser.get_outputs()  # to ensure the outputs are generated and saved correctly, even though we don't use them here

        nib.save(nib.Nifti1Image(volume.astype(np.float32), self.parser.coronary_affine), volume_path)
        np.savez_compressed(point_cloud_path, xyz=outputs.point_cloud.xyz.astype(np.float32))

        self.assertTrue(volume_path.exists())
        self.assertTrue(point_cloud_path.exists())
        self.assertEqual(volume.shape, tuple(self.parser.volume_shape))
        self.assertGreater(outputs.point_cloud.xyz.shape[0], 0)


if __name__ == "__main__":
    unittest.main()