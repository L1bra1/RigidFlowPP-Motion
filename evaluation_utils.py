"""
Codes to evaluate motion prediction
Some of the code are modified based on 'eval.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch



def evaluate_motion_prediction(disp_pred, all_disp_field_gt, all_valid_pixel_maps, future_steps,
                               distance_intervals, selected_future_sweeps, cell_groups):

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(all_disp_field_gt.size(0), -1, pred_shape[-3], pred_shape[-2], pred_shape[-1])
    disp_pred = disp_pred.contiguous()
    disp_pred = disp_pred.cpu().numpy()


    # For those with very small movements, we consider them as static
    last_pred = disp_pred[:, -1, :, :, :]
    last_pred_norm = np.linalg.norm(last_pred, ord=2, axis=1)  # out: (batch, h, w)
    thd_mask = last_pred_norm <= 0.2

    # cat_weight_map = np.ones_like(FGBG_pred_numpy, dtype=np.float32)
    cat_weight_map = np.ones_like(last_pred_norm, dtype=np.float32)
    cat_weight_map[thd_mask] = 0.0
    cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)

    disp_pred = disp_pred * cat_weight_map  # small motion, static, background


    # Pre-processing
    all_disp_field_gt = all_disp_field_gt.numpy()  # (bs, seq, h, w, channel)
    future_steps = future_steps.numpy()[0]

    valid_pixel_maps = all_valid_pixel_maps[:, -future_steps:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.numpy()

    all_disp_field_gt = all_disp_field_gt[:, -future_steps:, ]
    all_disp_field_gt = np.transpose(all_disp_field_gt, (0, 1, 4, 2, 3))
    all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)

    upper_thresh = 0.2
    upper_bound = 1 / 20 * upper_thresh

    static_cell_mask = all_disp_field_gt_norm <= upper_bound
    static_cell_mask = np.all(static_cell_mask, axis=1)  # along the temporal axis
    moving_cell_mask = np.logical_not(static_cell_mask)

    for j, d in enumerate(distance_intervals):
        for slot, s in enumerate((selected_future_sweeps - 1)):  # selected_future_sweeps: [4, 8, ...]
            curr_valid_pixel_map = valid_pixel_maps[:, s]

            if j == 0:  # corresponds to static cells
                curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
            else:
                # We use the displacement between keyframe and the last sample frame as metrics
                last_gt_norm = all_disp_field_gt_norm[:, -1]
                mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])

                curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                curr_mask = np.logical_and(curr_mask, moving_cell_mask)

            # we evaluate the performance for cells within the range [-30m, 30m] along both x, y dimensions.
            border = 8
            roi_mask = np.zeros_like(curr_mask, dtype=np.bool_)
            roi_mask[:, border:-border, border:-border] = True
            curr_mask = np.logical_and(curr_mask, roi_mask)

            cell_idx = np.where(curr_mask == True)

            gt = all_disp_field_gt[:, s]
            pred = disp_pred[:, -1, :, :, :]
            norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)

            cell_groups[j][slot].append(norm_error[cell_idx])

    return cell_groups