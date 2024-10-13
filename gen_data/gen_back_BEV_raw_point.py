"""
This code is to generate raw point clouds, backward BEV maps, and information for the training samples in nuScenes data.
And the code is modified based on 'gen_data.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

from gen_data.nuscenes.nuscenes import NuScenes
import os
from gen_data.nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
from functools import reduce

from gen_data.nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/path_to/nuScenes/nuScenes-data/', type=str, help='Root path to nuScenes dataset')
parser.add_argument('-s', '--split', default='train', type=str, help='The data split [train/val/test]')
parser.add_argument('-p', '--savepath', default='/path_to/nuScenes/self-data/', type=str, help='Directory for saving the generated data')
parser.add_argument('--seed', default=1234, type=int)

args = parser.parse_args()

np.random.seed(args.seed)

nusc = NuScenes(version='v1.0-trainval', dataroot=args.root, verbose=True)
print("Total number of scenes:", len(nusc.scene))

class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
             'vehicle.bicycle': 3}  # background: 0, other: 4


if args.split == 'train':
    num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
    nsweeps_back = 30  # Number of frames back to the history (including the current timestamp)
    nsweeps_forward = 20  # Number of frames into the future (does not include the current timestamp)
    skip_frame = 0  # The number of frames skipped for the adjacent sequence
    num_adj_seqs = 2  # number of adjacent sequences, among which the time gap is \delta t
else:
    num_keyframe_skipped = 1
    nsweeps_back = 25  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
    nsweeps_forward = 20
    skip_frame = 0
    num_adj_seqs = 1


# The specifications for BEV maps
voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
num_past_frames_for_bev_seq = 5  # the number of past frames for BEV map sequence


scenes = np.load('split.npy', allow_pickle=True).item().get(args.split)
print("Split: {}, which contains {} scenes.".format(args.split, len(scenes)))

args.savepath = check_folder(args.savepath)
args.save_back_path = check_folder(os.path.join(args.savepath, 'motionnet-data-back'))
args.save_raw_pc_path = check_folder(os.path.join(args.savepath, 'raw-pc'))
args.save_info_path = check_folder(os.path.join(args.savepath, 'sample-info'))


def gen_data():
    res_scenes = list()
    for s in scenes:
        s_id = s.split('_')[1]
        res_scenes.append(int(s_id))

    for scene_idx in res_scenes:
        curr_scene = nusc.scene[scene_idx]

        first_sample_token = curr_scene['first_sample_token']
        curr_sample = nusc.get('sample', first_sample_token)
        curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])


        adj_seq_cnt = 0
        save_seq_cnt = 0  # only used for save data file name

        save_info_dict_list = list()
        saved_token_list = list()

        # Iterate each sample data
        print("Processing scene {} ...".format(scene_idx))
        while curr_sample_data['next'] != '':

            all_times = \
                LidarPointCloud.from_file_multisweep_bf_sample_data_return_times(nusc, curr_sample_data,
                                                                    nsweeps_back=nsweeps_back,
                                                                    nsweeps_forward=nsweeps_forward)

            _, sort_idx = np.unique(all_times, return_index=True)
            unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
            num_sweeps = len(unique_times)

            # Make sure we have sufficient past and future sweeps
            if num_sweeps != (nsweeps_back + nsweeps_forward):

                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

                # Reset
                adj_seq_cnt = 0
                save_info_dict_list = list()
                continue


            if adj_seq_cnt == 0:

                save_info_dict = dict()

                lidar_curr_sample = curr_sample.copy()
                key_timestamps = np.zeros(2)

                lidar_sd_token_data = nusc.get('sample_data', lidar_curr_sample['data']['LIDAR_TOP'])
                _, ref_from_car, car_from_global, ref_time = get_pc_pose(lidar_sd_token_data, inverse=True)

                # relocate to -1s
                for key_frame_index in range(2):
                    save_raw_pc_list = dict()

                    lidar_sd_token_data = nusc.get('sample_data', lidar_curr_sample['data']['LIDAR_TOP'])
                    lidar_sd_token = lidar_sd_token_data['token']
                    save_info_dict['token_' + str(key_frame_index)] = lidar_sd_token

                    key_timestamps[key_frame_index] = 1e-6 * lidar_sd_token_data['timestamp']

                    if lidar_sd_token not in saved_token_list:
                        ''' new PC, have to save'''

                        # raw PC
                        current_pc, car_from_frame, global_from_car, timestamp = get_pc_pose(lidar_sd_token_data, inverse=False)
                        _, frame_from_car, car_from_global, _ = get_pc_pose(lidar_sd_token_data, inverse=True)


                        ''' save pc, label, and trans_matrix '''
                        save_raw_pc_list['pc'] = current_pc
                        save_raw_pc_list['timestamp'] = timestamp

                        save_raw_pc_list['car_from_frame'] = car_from_frame
                        save_raw_pc_list['global_from_car'] = global_from_car

                        save_raw_pc_list['frame_from_car'] = frame_from_car
                        save_raw_pc_list['car_from_global'] = car_from_global

                        save_file_name = os.path.join(args.save_raw_pc_path, lidar_sd_token + '.npy')
                        np.save(save_file_name, arr=save_raw_pc_list)

                        saved_token_list.append(lidar_sd_token)

                    if key_frame_index != 1:
                        lidar_curr_sample = nusc.get('sample', lidar_curr_sample['next'])

                save_info_dict['key_timestamp'] = key_timestamps

                ''' save information'''
                save_info_dict_list.append(save_info_dict)

                ''' save back BEV; direct flip, the reference frame is the next frame'''
                lidar_next_sample = nusc.get('sample', curr_sample['next'])
                lidar_sd_token_data = nusc.get('sample_data', lidar_next_sample['data']['LIDAR_TOP'])  # get information
                pc_data_dict = gen_back_pc(nusc, lidar_sd_token_data)
                valid_flag = 1
                if len(pc_data_dict) != 17:# {0s ~ 0.8s}
                    valid_flag = 0
                else:
                    save_back_BEV_dict = generate_back_BEV(pc_data_dict, voxel_size, area_extents)


            adj_seq_cnt += 1
            if adj_seq_cnt == num_adj_seqs:

                for seq_idx, seq_weak_dict in enumerate(save_info_dict_list):
                    # save the data
                    save_directory = check_folder(os.path.join(args.save_info_path, str(scene_idx) + '_' + str(save_seq_cnt)))
                    save_file_name = os.path.join(save_directory, str(seq_idx) + '.npy')
                    np.save(save_file_name, arr=seq_weak_dict)

                    if valid_flag == 1:
                        save_directory = check_folder(os.path.join(args.save_back_path, str(scene_idx) + '_' + str(save_seq_cnt)))
                        save_file_name = os.path.join(save_directory, str(seq_idx) + '.npy')
                        np.save(save_file_name, arr=save_back_BEV_dict)
                    else:
                        print(str(scene_idx) + '_' + str(save_seq_cnt))

                    print("  >> {} - {} Finish sample: {}, sequence {}".format(seq_weak_dict['key_timestamp'][0], seq_weak_dict['key_timestamp'][1], save_seq_cnt, seq_idx))

                save_seq_cnt += 1
                adj_seq_cnt = 0
                save_info_dict_list = list()


                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            else:
                flag = False
                for _ in range(skip_frame + 1):
                    if curr_sample_data['next'] != '':
                        curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more sample frames
                    break




def get_pc_pose(ref_sd_rec, inverse = True):
    # Get reference pose and timestamp
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']),
                                    inverse=inverse)

    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=inverse)

    scan = np.fromfile((os.path.join(nusc.dataroot, ref_sd_rec['filename'])), dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]

    return points.T, ref_from_car, car_from_global, ref_time



def gen_back_pc(nusc, curr_sample_data):
    # 0{0}, 0.2{4}, 0.4{8}, 0.6{12}, 0.8{16},
    # current "lidar_curr_sample" is the 0{0}

    # in this code, we do not remove the close points
    all_pc, all_times, trans_matrices = \
        LidarPointCloud.from_file_multisweep_bf_sample_data(nusc, curr_sample_data,
                                                            return_trans_matrix=True,
                                                            nsweeps_back=1,
                                                            nsweeps_forward=16)

    # Store point cloud of each sweep
    pc = all_pc.points
    _, sort_idx = np.unique(all_times, return_index=True)
    unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
    num_sweeps = len(unique_times)

    pc_data_dict = dict()
    for tid in range(num_sweeps):
        _time = unique_times[tid]
        points_idx = np.where(all_times == _time)[0]
        _pc = pc[:, points_idx]
        pc_data_dict['pc_' + str(tid)] = _pc

    return pc_data_dict


def voxelize_occupy(pts, voxel_size, extents=None, return_indices=False):
    """
    Voxelize the input point cloud. We only record if a given voxel is occupied or not, which is just binary indicator.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param return_indices: Whether to return the non-empty voxel indices.
    """
    # Function Constants
    VOXEL_EMPTY = 0
    VOXEL_FILLED = 1

    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Create Voxel Object with -1 as empty/occluded
    leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)

    # Fill out the leaf layout
    leaf_layout[voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2]] = VOXEL_FILLED

    if return_indices:
        return leaf_layout, voxel_indices
    else:
        return leaf_layout

def generate_back_BEV(pc_data_dict, voxel_size, area_extents):

    # remove close pc
    pc_list = []
    sampled_index = [0, 4, 8, 12, 16]
    sampled_index = sampled_index[::-1]

    for i in sampled_index:
        pc = pc_data_dict['pc_' + str(i)].T
        pc_list.append(pc)

    # generate BEV
    voxel_indices_list = list()
    padded_voxel_points_list = list()
    for i in range(5):
        res, voxel_indices = voxelize_occupy(pc_list[i], voxel_size=voxel_size, extents=area_extents, return_indices=True)
        voxel_indices_list.append(voxel_indices)
        padded_voxel_points_list.append(res)

    # save BEV
    save_data_dict = dict()
    for i in range(len(voxel_indices_list)):
        save_data_dict['voxel_indices_' + str(i)] = voxel_indices_list[i].astype(np.int32)

    return save_data_dict

if __name__ == "__main__":
    gen_data()

