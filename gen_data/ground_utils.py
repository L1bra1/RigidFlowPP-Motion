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

def get_voxel_from_point(pc, voxel_size, extents):
    discrete_pc = np.floor(pc[:, :3] / voxel_size).astype(np.int32)
    min_voxel_coord = np.floor(extents.T[0] / voxel_size)
    max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1

    voxel_indices = (discrete_pc - min_voxel_coord).astype(int)

    return voxel_indices

def generate_ground_points(pc, raw_pc=None, distance_threshold=0.4, ransac_n=3, num_iterations=2000, height_threshold=-1, plane_model=None, angle_threshold_degrees=10):
    if plane_model is None:
        tmp = pc[:, 2] < height_threshold
        pc_low_height = pc[tmp]

        plane_model, _ = ransac_segment_horizontal_plane(pc_low_height, distance_threshold, ransac_n,
                                                           num_iterations, angle_threshold_degrees)

    if raw_pc is None:
        raw_pc = pc
    distances = np.abs(np.dot(raw_pc, plane_model[:3]) + plane_model[3]) / np.linalg.norm(plane_model[:3])
    inliers = np.where(distances < distance_threshold)[0]
    outliers = np.where(distances >= distance_threshold)[0]

    ground_pc = raw_pc[inliers]
    non_ground_pc = raw_pc[outliers]

    ground_mask = np.zeros([raw_pc.shape[0]], dtype=np.bool)
    ground_mask[inliers] = 1
    return ground_pc, non_ground_pc, ground_mask, plane_model



def fit_plane(points):
# Define the fit_plane function: This function takes a set of points and returns the coefficients
# (A, B, C, D) of the best-fitting plane in the form Ax + By + Cz + D = 0.

    centroid = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]
    normal = normal[:3] / np.linalg.norm(normal[:3])
    d = -np.dot(centroid, normal)


    # augmented_points = np.column_stack((points, np.ones(len(points)))).astype(np.float32)
    # _, _, vh = np.linalg.svd(augmented_points)
    # solution_vector = vh[-1, :]
    # normal_vector = solution_vector[:3] / np.linalg.norm(solution_vector[:3])
    # d_normalized = solution_vector[-1] / np.linalg.norm(solution_vector[:3])

    return np.append(normal, d)


def is_plane_horizontal(plane_model, horizontal_axis=2, angle_threshold_degrees=30):
    normal = plane_model[:3]
    reference_vector = np.zeros(3)
    reference_vector[horizontal_axis] = 1

    angle_radians = np.arccos(np.dot(normal, reference_vector) / (np.linalg.norm(normal) * np.linalg.norm(reference_vector)))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees <= angle_threshold_degrees or angle_degrees >= (180-angle_threshold_degrees)

def ransac_segment_horizontal_plane(points, distance_threshold=0.1, ransac_n=3, num_iterations=1000,angle_threshold_degrees=30):
    best_inliers = []
    best_plane = None

    for _ in range(num_iterations):
        sampled_points = points[np.random.choice(points.shape[0], ransac_n, replace=False)]
        candidate_plane = fit_plane(sampled_points)

        if not is_plane_horizontal(candidate_plane, angle_threshold_degrees=angle_threshold_degrees):
            continue

        distances = np.abs(np.dot(points, candidate_plane[:3]) + candidate_plane[3]) / np.linalg.norm(candidate_plane[:3])
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = candidate_plane

    return best_plane, best_inliers
