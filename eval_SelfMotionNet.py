"""
Evaluate self-supervised MotionNet
Some of the code are modified based on 'train_single_seq.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from self_model import SelfMotionNet
from data.self_nuscenes_dataloader import DatasetSingleSeq_Self

from tqdm import tqdm

from pseudo_labels_utils.motion_loss import Motion_Loss_Module
# from loss_utils_for_self import CCD_loss_for_self

from evaluation_utils import evaluate_motion_prediction


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

out_seq_len = 1  # The number of future frames we are going to predict
height_feat_size = 13  # The size along the height dimension

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--evaldata', default='/path_to/nuScenes/input-data/test', type=str, help='The path to the preprocessed sparse BEV training data')

parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--batch', default=1, type=int, help='Batch size')
parser.add_argument('--nworker', default=1, type=int, help='Number of workers')
parser.add_argument('--log', default=True, action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')
parser.add_argument('--gpu', default='0')
parser.add_argument('--pretrained', default='pretrained/model_nuScenes.pth', type=str)

args = parser.parse_args()
print(args)

need_log = args.log
BATCH_SIZE = args.batch
num_workers = args.nworker
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'train_single_seq'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()


        else:
            model_save_path = args.resume  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])


    evalset = DatasetSingleSeq_Self(dataset_root=args.evaldata, split='test', future_frame_skip=0,
                                    voxel_size=voxel_size, area_extents=area_extents)


    evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=1)
    print("Val dataset size:", len(evalset))


    model = SelfMotionNet(out_seq_len=out_seq_len, height_feat_size=height_feat_size)
    model = nn.DataParallel(model)
    model = model.to(device)

    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print("Load model from {}".format(args.pretrained))


    model.eval()
    me_0, me_5, me_20= eval(model, evalloader, device)

    if need_log:
        saver.write("{}\t{}\t{}\n".format(me_0, me_5, me_20))
        saver.flush()

    if need_log:
        saver.close()



def eval(model, evalloader, device):

    # Motion
    num_future_sweeps = 20
    frequency = 20.0
    speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])

    selected_future_sweeps = np.arange(0, num_future_sweeps + 1, num_future_sweeps)  # We evaluate predictions at 1s
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / frequency)  # "20" is because the LIDAR scanner is 20Hz

    cell_groups = list()  # grouping the cells with different speeds
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(len(selected_future_sweeps)):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)

    # for i, data in enumerate(evalloader, 0):
    for i, data in tqdm(enumerate(evalloader, 0), total=len(evalloader), smoothing=0.9):
        padded_voxel_points, _, \
        all_disp_field_gt, non_empty_map, \
        all_valid_pixel_maps, future_steps, _, \
        _, _, _, _, _, _ = data

        padded_voxel_points = padded_voxel_points.to(device)

        with torch.no_grad():
            disp_pred = model(padded_voxel_points)
            disp_pred = disp_pred * 2.0


            cell_groups = evaluate_motion_prediction(disp_pred, all_disp_field_gt, all_valid_pixel_maps,
                                                     future_steps, distance_intervals,
                                                     selected_future_sweeps, cell_groups)

    me_list = np.zeros([3])
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]
        print("--------------------------------------------------------------")
        print("For cells within speed range [{}, {}]:\n".format(d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(len(selected_future_sweeps)):
            row = group[s]

            errors = np.concatenate(row) if len(row) != 0 else row

            if len(errors) == 0:
                mean_error = None
                error_quantile_50 = None
            else:
                mean_error = np.average(errors)
                error_quantile_50 = np.quantile(errors, 0.5)

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)

            msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}". \
                format(selected_future_sweeps[s], mean_error, error_quantile_50)
            print(msg)
        me_list[i] = mean_error


    return me_list[0], me_list[1], me_list[2]

if __name__ == "__main__":
    main()
