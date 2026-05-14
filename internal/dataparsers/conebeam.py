from dataclasses import dataclass
import numpy as np

@dataclass
class ConeBeamParams:
    nVoxels: np.ndarray  # (3,) array of number of voxels in each dimension (nx, ny, nz)
    min_pt_world: np.ndarray  # (3,) array of minimum coordinates of the volume in world space
    max_pt_world: np.ndarray  # (3,) array of maximum coordinates of the volume
    alphas: np.ndarray  # (num_proj,) array of projection angles in radians
    nh: int  # number of detector pixels in height
    nw: int  # number of detector pixels in width
    sh: float  # physical size of the detector in height
    sw: float  # physical size of the detector in width
    dde: float  # distance from origin to detector center
    dso: float  # distance from origin to source


def build_conebeam_geometry(
    params: ConeBeamParams,
    impl: Literal["astra_cuda", "astra_cpu"] = "astra_cuda",
):
    min_pt = [float(params.min_pt_world[0]), float(params.min_pt_world[1]), float(params.min_pt_world[2])]
    max_pt = [float(params.max_pt_world[0]), float(params.max_pt_world[1]), float(params.max_pt_world[2])]

    reco_space = odl.uniform_discr(
        min_pt=min_pt,
        max_pt=max_pt,
        shape=[int(params.nVoxels[0]), int(params.nVoxels[1]), int(params.nVoxels[2])],
        dtype="float32",
    )

    angle_partition = odl.nonuniform_partition(params.alphas.astype(np.float32))

    detector_partition = odl.uniform_partition(
        min_pt=[-(params.sh / 2.0), -(params.sw / 2.0)],
        max_pt=[(params.sh / 2.0), (params.sw / 2.0)],
        shape=[int(params.nh), int(params.nw)],
        nodes_on_bdry=True,
    )

    geometry = odl.tomo.ConeBeamGeometry(
        apart=angle_partition,
        dpart=detector_partition,
        src_radius=float(params.dso),
        det_radius=float(params.dde),
        axis=[0, 0, 1],
    )

    ray_trafo = odl.tomo.RayTransform(
        vol_space=reco_space,
        geometry=geometry,
        impl=impl,
    )

    fbp_op = odl.tomo.fbp_op(
        ray_trafo=ray_trafo,
        filter_type="Ram-Lak",
        frequency_scaling=1.0,
    )

    return reco_space, ray_trafo, fbp_op