import json
from pathlib import Path

import numpy as np
import pytest

from internal.dataparsers.xray_dataparser.meta import XRayMetaLoader


def test_xray_meta_loader_loads_meta(tmp_path: Path):
	meta = {
		"coronary_type": "LCA",
		"num_frames": 2,
		"volume_size": [1, 2, 3],
		"centering_affine_dict": {
			"LCA": [
				[1, 0, 0, 10],
				[0, 1, 0, 20],
				[0, 0, 1, 30],
				[0, 0, 0, 1],
			],
			"RCA": [
				[2, 0, 0, 0],
				[0, 2, 0, 0],
				[0, 0, 2, 0],
				[0, 0, 0, 1],
			],
		},
		"c_arm_geometry": {
			"sdd": 1000,
			"sod": 800,
			"height": 1024,
			"width": 1024,
			"delx": 0.2,
			"dely": 0.2,
			"x0": 0.0,
			"y0": 0.0,
		},
		"rotated_parameters": {
			"total_frame": 2,
			"alpha_start": 10.0,
			"beta_start": 5.0,
			"angular_velocity": 3.0,
			"fps": 30.0,
			"coordinate_system": "RAS",
			"parameterization": "euler_angles",
			"convention": "ZXY",
		},
		"frames": [
			{"frame": 0, "time_s": 0.0, "phase": 0.0, "alpha_degree": 10.0, "beta_degree": 5.0},
			{"frame": 1, "time_s": 0.033, "phase": 0.5, "alpha_degree": 9.9, "beta_degree": 5.0},
		],
	}
	(tmp_path / "rotate_dsa.json").write_text(json.dumps(meta))

	result = XRayMetaLoader().load(tmp_path)

	assert result.coronary_type == "LCA"
	assert result.num_frames == 2
	assert np.array_equal(result.volume_size, np.array([1.0, 2.0, 3.0]))
	assert np.array_equal(result.centering_affine, np.array(meta["centering_affine_dict"]["LCA"], dtype=np.float64))
	assert np.array_equal(result.volume_origin, np.array([10.0, 20.0, 30.0]))
	assert np.array_equal(
		result.centering_affine_dict["RCA"],
		np.array(meta["centering_affine_dict"]["RCA"], dtype=np.float64),
	)
	assert result.c_arm_geometry.sdd == 1000
	assert result.rotated_parameters.total_frame == 2
	assert np.allclose(result.alphas_radians, np.deg2rad([10.0, 9.9]))
	assert np.allclose(result.phase_array, np.array([0.0, 0.5]))
	assert np.allclose(result.time_array, np.array([0.0, 0.033]))
	assert len(result.frames) == 2
	assert result.frames[0].frame == 0


def test_xray_meta_loader_supports_legacy_keys(tmp_path: Path):
	meta = {
		"coronary_type": "RCA",
		"volume_size": [1, 2, 3],
		"lca_centering_affine": [
			[1, 0, 0, 10],
			[0, 1, 0, 20],
			[0, 0, 1, 30],
			[0, 0, 0, 1],
		],
		"rca_centering_affine": [
			[2, 0, 0, -10],
			[0, 2, 0, -20],
			[0, 0, 2, -30],
			[0, 0, 0, 1],
		],
		"c_arm_geometry": {
			"sdd": 900,
			"sod": 700,
			"height": 512,
			"width": 512,
			"delx": 0.3,
			"dely": 0.3,
			"x0": 2.0,
			"y0": 4.0,
		},
		"rotate_parameters": {
			"total_frames": 3,
			"alpha_start": 0.0,
			"beta_start": 0.0,
			"angular_velocity": 0.0,
			"fps": 25.0,
		},
		"frames": [
			{"frame": 0, "time_s": 0.0, "phase": 0.0, "alpha_degree": 0.0, "beta_degree": 0.0},
			{"frame": 1, "time_s": 0.04, "phase": 0.25, "alpha_degree": 0.0, "beta_degree": 0.0},
			{"frame": 2, "time_s": 0.08, "phase": 0.5, "alpha_degree": 0.0, "beta_degree": 0.0},
		],
	}
	(tmp_path / "rotate_dsa.json").write_text(json.dumps(meta))

	result = XRayMetaLoader().load(tmp_path)

	assert result.coronary_type == "RCA"
	assert result.num_frames == 3
	assert result.rotated_parameters.total_frame == 3
	assert np.array_equal(result.centering_affine, np.array(meta["rca_centering_affine"], dtype=np.float64))


def test_xray_meta_loader_missing_file(tmp_path: Path):
	with pytest.raises(FileNotFoundError):
		XRayMetaLoader().load(tmp_path)
