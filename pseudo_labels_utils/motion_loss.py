import numpy as np
import torch

from sp_utils import compute_sp
from pseudo_labels_utils.Conf_aware_Rigid_iter_utils import Conf_aware_Rigid_Reg

class Motion_Loss_Module(torch.nn.Module):
    def __init__(self, iter=2, theta0=1.0, bg_weight=0.05, n_sp=60, flow_thr=3.0, pos_thr=1.0):
        super(Motion_Loss_Module, self).__init__()
        self.iter = iter
        self.theta0 = theta0
        self.bg_weight = bg_weight
        self.n_sp = n_sp
        self.flow_thr = flow_thr
        self.pos_thr = pos_thr

    def forward(self, disp_pred, back_disp_pred, source_pc, target_pc, ref_pc, source_num, target_num, voxel_size, extents, padded_voxel_points,
                          BEV_ground, conf_aware):

        non_empty_map = (torch.sum(padded_voxel_points[:, -1, :, :, :], -1) > 0).float()

        device = non_empty_map.device
        loss_disp = torch.zeros((1), device=disp_pred.device, dtype=disp_pred.dtype)

        batch_size = disp_pred.shape[0]
        for batch_index in range(batch_size):
            curr_valid_map = non_empty_map[batch_index]

            # get source, target pc, predicted point flow
            curr_source_num = source_num[batch_index]
            curr_target_num = target_num[batch_index]

            curr_source_pc_np = source_pc[batch_index, :curr_source_num, :].cpu().numpy()
            curr_target_pc_np = target_pc[batch_index, :curr_target_num, :].cpu().numpy()
            curr_disp_pred = disp_pred[batch_index, :, :, :]

            curr_ref_pc_np = ref_pc[batch_index, :curr_target_num, :].cpu().numpy()
            curr_back_disp_pred = back_disp_pred[batch_index, :, :, :]

            # get source static pc
            curr_BEV_ground_source = BEV_ground[batch_index].float()

            # get predicted point flow, valid for each point
            curr_voxel_indices = gen_voxel_indices_for_pc(curr_source_pc_np, voxel_size, extents)
            curr_point_disp_pred = curr_disp_pred[:, curr_voxel_indices[:, 0], curr_voxel_indices[:, 1]].permute(1, 0)

            curr_reference_voxel_indices = gen_voxel_indices_for_pc(curr_ref_pc_np, voxel_size, extents)
            curr_point_back_disp_pred = curr_back_disp_pred[:, curr_reference_voxel_indices[:, 0],
                                        curr_reference_voxel_indices[:, 1]].permute(1, 0)

            # get fg mask, bg mask
            curr_fg_map = torch.zeros_like(curr_valid_map)
            curr_fg_map[curr_voxel_indices[:, 0], curr_voxel_indices[:, 1]] = 1
            curr_fg_map = curr_fg_map * curr_valid_map
            num_voxel_fg = torch.sum(curr_fg_map)

            curr_bg_map = curr_BEV_ground_source * curr_valid_map
            num_voxel_bg = torch.sum(curr_bg_map)

            # get warped source pc
            curr_source_pc = torch.from_numpy(curr_source_pc_np).to(device).float()
            curr_target_pc = torch.from_numpy(curr_target_pc_np).to(device).float()
            curr_point_3d_disp_pred = torch.cat([curr_point_disp_pred, torch.zeros_like(curr_point_disp_pred[:, 0:1])], -1)
            curr_point_3d_back_disp_pred = torch.cat([curr_point_back_disp_pred, torch.zeros_like(curr_point_back_disp_pred[:, 0:1])], -1)

            ''' moving part'''
            curr_source_pc_np = curr_source_pc.cpu().numpy()
            voxel_label = compute_sp(curr_source_pc_np, self.n_sp)
            voxel_label = torch.from_numpy(voxel_label).cuda().int().unsqueeze(0)

            pseudo_gt, label_validity_mask, pos_diff, flow_diff = Conf_aware_Rigid_Reg(curr_source_pc.unsqueeze(0),
                                                                                       curr_point_3d_disp_pred.unsqueeze(0),
                                                                                       curr_point_3d_back_disp_pred.unsqueeze(0),
                                                                                       voxel_label, curr_target_pc.unsqueeze(0),
                                                                                       conf_aware, iter=self.iter, theta0=self.theta0,
                                                                                       flow_thr=self.flow_thr, pos_thr=self.pos_thr)
            if conf_aware:
                validity_mask = label_validity_mask * (flow_diff < self.flow_thr).float() * (pos_diff < self.pos_thr).float()
            else:
                validity_mask = label_validity_mask

            pseudo_gt = pseudo_gt.squeeze(0)
            validity_mask = validity_mask.squeeze(0)

            fg_loss_per_voxel = torch.sum(torch.abs(pseudo_gt - curr_point_3d_disp_pred), -1)
            fg_loss_per_voxel = torch.sum(fg_loss_per_voxel * validity_mask) / (torch.sum(validity_mask) + 1e-6)

            # ground loss
            bg_gt = torch.zeros_like(curr_disp_pred)
            bg_loss_per_voxel = torch.sum(torch.abs(curr_disp_pred * curr_bg_map.unsqueeze(0) - bg_gt * curr_bg_map.unsqueeze(0)), 0)
            bg_loss_per_voxel = torch.sum(bg_loss_per_voxel) / (torch.sum(curr_bg_map) + 1e-6)

            # weighted sum loss
            curr_loss = (fg_loss_per_voxel * num_voxel_fg + self.bg_weight * bg_loss_per_voxel * num_voxel_bg) \
                        / (num_voxel_fg + num_voxel_bg + 1e-6)

            loss_disp = loss_disp + curr_loss

        loss = loss_disp / batch_size
        return loss


def gen_voxel_indices_for_pc(pc, voxel_size, extents):
    discrete_pc = np.floor(pc[:, :3] / voxel_size).astype(np.int32)
    min_voxel_coord = np.floor(extents.T[0] / voxel_size)
    voxel_indices = (discrete_pc - min_voxel_coord).astype(int)
    return voxel_indices

