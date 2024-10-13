import numpy as np
import os

def remove_close(points, radius):
    points = points.T
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    points = points.T
    return points, not_close

def filter_pc(pc, extents):
    filter_idx = np.where((extents[0, 0] < pc[:, 0]) & (pc[:, 0] < extents[0, 1]) &
                          (extents[1, 0] < pc[:, 1]) & (pc[:, 1] < extents[1, 1]) &
                          (extents[2, 0] < pc[:, 2]) & (pc[:, 2] < extents[2, 1]))[0]
    pc = pc[filter_idx]
    return pc, filter_idx

def gen_voxel_indices_for_pc(pc, voxel_size, extents):
    # Convert 3D coordinate to voxel index
    discrete_pc = np.floor(pc[:, :3] / voxel_size).astype(np.int32)
    min_voxel_coord = np.floor(extents.T[0] / voxel_size)
    voxel_indices = (discrete_pc - min_voxel_coord).astype(int)
    range = np.floor((extents.T[1] - extents.T[0])/ voxel_size) - 1

    voxel_indices[voxel_indices[:, 0] > range[0], 0] = range[0]
    voxel_indices[voxel_indices[:, 1] > range[1], 1] = range[1]
    voxel_indices[voxel_indices[:, 2] > range[2], 2] = range[2]

    return voxel_indices


def generate_BEV_ground_mask(ground_pc, non_ground_pc, voxel_size, area_extents):
    # BEV_ground
    voxel_indices = gen_voxel_indices_for_pc(ground_pc, voxel_size, area_extents)
    BEV_ground = np.zeros([256, 256], dtype=np.bool_)
    BEV_ground[voxel_indices[:, 0], voxel_indices[:, 1]] = 1

    # BEV_non_ground
    voxel_indices = gen_voxel_indices_for_pc(non_ground_pc, voxel_size, area_extents)
    BEV_non_ground = np.zeros([256, 256], dtype=np.bool_)
    BEV_non_ground[voxel_indices[:, 0], voxel_indices[:, 1]] = 1

    BEV_ground_2 = BEV_ground * (1 - BEV_non_ground)  # final Ground map

    return BEV_ground_2.astype(np.bool_)



def sample_non_ground_point(pc, plane_model, voxel_size, area_extents,
                            distance_threshold=0.4, need_BEV=False, need_refer=False):

    pc, not_close = remove_close(pc, radius=1.0)
    pc, filter_idx = filter_pc(pc, extents=area_extents)

    if plane_model is not None:
        distances = np.abs(np.dot(pc, plane_model[:3]) + plane_model[3]) / np.linalg.norm(plane_model[:3])
        inliers = np.where(distances < distance_threshold)[0]
        ground_mask = np.zeros([pc.shape[0]], dtype=np.bool_)
        ground_mask[inliers] = 1
        FG_point = pc[~ground_mask]
    else:
        FG_point = pc

    if need_BEV:
        if plane_model is not None:
            BG_point = pc[ground_mask]
            BEV_ground = generate_BEV_ground_mask(BG_point, FG_point, voxel_size, area_extents)
        else:
            BEV_ground = np.zeros([256, 256], dtype=np.bool_)
    else:
        BEV_ground = None

    return FG_point, BEV_ground



def sample_fixed_num_point(pc, num_points_motion):

    FG_point = pc
    FG_point_num = FG_point.shape[0]

    if FG_point_num != 0:
        if FG_point_num >= num_points_motion:
            sample_idx = np.random.choice(FG_point_num, num_points_motion, replace=False)
            FG_point_num = num_points_motion
        else:
            sample_idx = np.concatenate((np.arange(FG_point_num),
                                         np.random.choice(FG_point_num, num_points_motion - FG_point_num,
                                                          replace=True)), axis=-1)
        FG_point = FG_point[sample_idx]
    else:
        FG_point = np.zeros((num_points_motion, 3))

    return FG_point, FG_point_num