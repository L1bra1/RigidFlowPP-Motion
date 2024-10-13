"""
Data loader for nuScenes data
Some of the code are modified based on 'nuscenes_dataloader.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

from torch.utils.data import Dataset
import numpy as np
import os
import warnings
from data.data_utils import remove_close, sample_non_ground_point, sample_fixed_num_point

from functools import reduce


class DatasetSingleSeq_Self(Dataset):
    def __init__(self, dataset_root=None, raw_pc_root=None, info_root=None, data_back_root=None,
                 raw_ground_root=None, split='train', future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-3., 2.]]), dims=(256, 256, 13), num_category=5,
                 num_points_motion=30000):

        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        print("data root:", dataset_root)
        self.raw_pc_root = raw_pc_root
        self.info_root = info_root
        self.data_back_root = data_back_root
        self.raw_ground_root = raw_ground_root

        seq_dirs = []
        if split == 'train':
            for d in os.listdir(self.data_back_root):
                tmp_0 = os.path.join(self.dataset_root, d) + '/0.npy'
                seq_dirs.append(tmp_0)
        else:
            for d in os.listdir(self.dataset_root):
                tmp_0 = os.path.join(self.dataset_root, d) + '/0.npy'
                seq_dirs.append(tmp_0)

        self.seq_files = seq_dirs
        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        # For training, the size of dataset should be 17025; for validation: 1719; for testing: 4309
        if split == 'train' and self.num_sample_seqs != 17025:
            warnings.warn(">> The size of training dataset is not 17025.\n")
        elif split == 'val' and self.num_sample_seqs != 1719:
            warnings.warn(">> The size of validation dataset is not 1719.\n")
        elif split == 'test' and self.num_sample_seqs != 4309:
            warnings.warn('>> The size of test dataset is not 4309.\n')

        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.future_frame_skip = future_frame_skip
        self.dims = dims
        self.num_points_motion = num_points_motion
        self.num_category = num_category

    def __len__(self):
        return self.num_sample_seqs


    def extract_pc_matrix(self, token):
        pc_file_name = os.path.join(self.raw_pc_root, token+'.npy')
        pc_data_handle = np.load(pc_file_name, allow_pickle=True)
        pc_dict = pc_data_handle.item()

        return pc_dict['pc'], pc_dict['car_from_frame'], pc_dict['global_from_car'],\
               pc_dict['frame_from_car'], pc_dict['car_from_global']

    def flip_op(self, x, y):
        tmp = y.copy()
        y = x.copy()
        x = tmp.copy()
        return x, y

    def __getitem__(self, idx):
        seq_file = self.seq_files[idx]
        gt_data_handle = np.load(seq_file, allow_pickle=True)
        gt_dict = gt_data_handle.item()

        dims = gt_dict['3d_dimension']
        num_future_pcs = gt_dict['num_future_pcs']
        num_past_pcs = gt_dict['num_past_pcs']
        pixel_indices = gt_dict['pixel_indices']

        sparse_disp_field_gt = gt_dict['disp_field']
        all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
        all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

        sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
        all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
        all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

        sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
        pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
        pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

        non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
        non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

        padded_voxel_points = list()
        for i in range(num_past_pcs):
            indices = gt_dict['voxel_indices_' + str(i)]
            curr_voxels = np.zeros(dims, dtype=np.bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

        # get self-supervision
        if self.split == 'train':
            scene_name = seq_file.split('/')[-2]

            info_file_name = os.path.join(os.path.join(self.info_root, scene_name), '0.npy')
            info_data_handle = np.load(info_file_name, allow_pickle=True)
            info_dict = info_data_handle.item()

            # load backward BEV
            BK_BEV_file_name = os.path.join(os.path.join(self.data_back_root, scene_name), '0.npy')
            BK_BEV_handle = np.load(BK_BEV_file_name, allow_pickle=True)
            BK_BEV_dict = BK_BEV_handle.item()

            BK_padded_voxel_points = list()
            for i in range(num_past_pcs):
                indices = BK_BEV_dict['voxel_indices_' + str(i)]
                curr_voxels = np.zeros(dims, dtype=np.bool_)
                curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                BK_padded_voxel_points.append(curr_voxels)
            BK_padded_voxel_points = np.stack(BK_padded_voxel_points, 0).astype(np.float32)

            ''' flip or not '''
            tmp = np.random.rand(1)
            if tmp > 0.5: # use flip; regard frame 1 as current frame{0}
                padded_voxel_points, BK_padded_voxel_points = self.flip_op(padded_voxel_points, BK_padded_voxel_points)
                token_0 = info_dict['token_1']
                token_1 = info_dict['token_0']
            else:
                token_0 = info_dict['token_0']
                token_1 = info_dict['token_1']


            # load ground information
            ground_info_token_0 = os.path.join(self.raw_ground_root, token_0 + '.npy')
            ground_info_token_0_handle = np.load(ground_info_token_0, allow_pickle=True)
            ground_info_token_0_dict = ground_info_token_0_handle.item()
            plane_model_0 = ground_info_token_0_dict['plane_model']

            ground_info_token_1 = os.path.join(self.raw_ground_root, token_1 + '.npy')
            ground_info_token_1_handle = np.load(ground_info_token_1, allow_pickle=True)
            ground_info_token_1_dict = ground_info_token_1_handle.item()
            plane_model_1 = ground_info_token_1_dict['plane_model']

            non_empty_map = (np.sum(padded_voxel_points[-1, :, :, :], -1) > 0).astype(np.float32)

            raw_pc0, car0_from_frame0, global_from_car0,\
            frame0_from_car0, car0_from_global = self.extract_pc_matrix(token_0)
            raw_pc0, not_close0 = remove_close(raw_pc0.T, radius=1.0)
            raw_pc0 = raw_pc0.T

            raw_pc1, car1_from_frame1, global_from_car1,\
            frame1_from_car1, car1_from_global = self.extract_pc_matrix(token_1)
            raw_pc1, not_close1 = remove_close(raw_pc1.T, radius=1.0)
            raw_pc1 = raw_pc1.T

            pc0 = raw_pc0[:3, :].T
            pc1 = raw_pc1[:3, :].T

            FG_point_0, BEV_ground = sample_non_ground_point(pc0, plane_model_0,
                                                             self.voxel_size, self.area_extents, need_BEV=True)
            FG_point_1, _ = sample_non_ground_point(pc1, plane_model_1,
                                                    self.voxel_size, self.area_extents, need_BEV=False)
            Ref_point = FG_point_1.copy()

            FG_point_1 = FG_point_1.T
            trans_matrix_1 = reduce(np.dot, [frame0_from_car0, car0_from_global, global_from_car1, car1_from_frame1])
            FG_point_1 = trans_matrix_1.dot(np.vstack((FG_point_1[:3, :], np.ones(FG_point_1.shape[1]))))[:3, :].T

            FG_point_0, FG_point_num_0 = sample_fixed_num_point(FG_point_0, self.num_points_motion)
            FG_point_1, FG_point_num_1 = sample_fixed_num_point(FG_point_1, self.num_points_motion)
            Ref_point, Ref_point_num = sample_fixed_num_point(Ref_point, self.num_points_motion)

        else:
            BEV_ground = np.zeros(1)
            FG_point_0 = np.zeros(1)
            FG_point_num_0 = np.zeros(1)
            FG_point_1 = np.zeros(1)
            FG_point_num_1 = np.zeros(1)
            Ref_point = np.zeros(1)
            Ref_point_num = np.zeros(1)
            BK_padded_voxel_points = np.zeros(1)


        return padded_voxel_points.astype(np.float32), BK_padded_voxel_points.astype(np.float32),\
               all_disp_field_gt.astype(np.float32), non_empty_map.astype(np.float32), \
               all_valid_pixel_maps.astype(np.float32), num_future_pcs, BEV_ground.astype(np.float32), \
               FG_point_0.astype(np.float32), FG_point_num_0, FG_point_1.astype(np.float32), FG_point_num_1,\
               Ref_point.astype(np.float32), Ref_point_num