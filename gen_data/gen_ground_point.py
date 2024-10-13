import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from tqdm import tqdm
from gen_data.ground_utils import remove_close, filter_pc, generate_ground_points
from functools import reduce
import os.path as osp
import glob

# import open3d as o3d

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--savepath', default='/path_to/nuScenes/self-data/', type=str, help='Directory for saving the generated data')
parser.add_argument('--seed', default=1234, type=int)

args = parser.parse_args()
np.random.seed(args.seed)

raw_pc_root = os.path.join(args.savepath, 'raw-pc')
info_root = os.path.join(args.savepath, 'sample-info')
ground_info_root = check_folder(os.path.join(args.savepath, 'ground-info'))

area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
voxel_size = (0.25, 0.25, 0.4)


def extract_pc_matrix(raw_pc_root, token):
    pc_file_name = os.path.join(raw_pc_root, token + '.npy')
    pc_data_handle = np.load(pc_file_name, allow_pickle=True)
    pc_dict = pc_data_handle.item()

    return pc_dict['pc'], pc_dict['car_from_frame'], pc_dict['global_from_car'], \
           pc_dict['frame_from_car'], pc_dict['car_from_global']

def process_pc(raw_pc_root, token_0, token_1):

    pc_dict = dict()

    raw_pc0, car0_from_frame0, global_from_car0, \
    frame0_from_car0, car0_from_global = extract_pc_matrix(raw_pc_root, token_0)
    _, not_close0 = remove_close(raw_pc0.T, radius=1.0) # current

    raw_pc1, car1_from_frame1, global_from_car1, \
    frame1_from_car1, car1_from_global = extract_pc_matrix(raw_pc_root, token_1)
    _, not_close1 = remove_close(raw_pc1.T, radius=1.0) # future

    trans_matrix_pc0_to_frame1 = reduce(np.dot, [frame1_from_car1, car1_from_global, global_from_car0, car0_from_frame0])
    pc0_in_frame1 = trans_matrix_pc0_to_frame1.dot(np.vstack((raw_pc0[:3, :], np.ones(raw_pc0.shape[1]))))[:3, :].T

    trans_matrix_pc1_to_frame0 = reduce(np.dot, [frame0_from_car0, car0_from_global, global_from_car1, car1_from_frame1])
    pc1_in_frame0 = trans_matrix_pc1_to_frame0.dot(np.vstack((raw_pc1[:3, :], np.ones(raw_pc1.shape[1]))))[:3, :].T

    pc_dict['pc0_in_frame0'] = raw_pc0[:3, :].T
    pc_dict['pc0_in_frame1'] = pc0_in_frame1

    pc_dict['pc1_in_frame0'] = pc1_in_frame0
    pc_dict['pc1_in_frame1'] = raw_pc1[:3, :].T

    pc_dict['pc0_not_close'] = not_close0
    pc_dict['pc1_not_close'] = not_close1

    return pc_dict


if __name__ == '__main__':

    seq_files = glob.glob(osp.join(raw_pc_root, "*.npy"))
    seq_files = np.sort(seq_files)#[0::2]

    for idx, file in tqdm(enumerate(seq_files, 0), total=len(seq_files), smoothing=0.9):

        file_name = file.split('/')[-1]

        pc_data_handle = np.load(file, allow_pickle=True)
        pc_dict = pc_data_handle.item()
        pc = pc_dict['pc'][:3, :].T


        ground_info_dict = dict()

        pc, not_close = remove_close(pc, radius=1.0)
        pc, filter_idx = filter_pc(pc, extents=area_extents)

        # generate ground information using plane fitting.
        ground_pc, non_ground_pc, ground_mask, plane_model = \
            generate_ground_points(pc, None, distance_threshold=0.4, ransac_n=3,
                                       num_iterations=2000, height_threshold=-1, angle_threshold_degrees=10)

        ground_info_dict['ground_mask'] = ground_mask
        ground_info_dict['plane_model'] = plane_model

        save_name = os.path.join(ground_info_root, file_name)
        np.save(save_name, arr=ground_info_dict)


