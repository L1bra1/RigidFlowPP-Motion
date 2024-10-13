# RigidFlow++ (motion prediction part)
This is the PyTorch code for [Self-Supervised 3D Scene Flow Estimation and Motion Prediction using Local Rigidity Prior (T-PAMI 2024)](https://www.computer.org/csdl/journal/tp/5555/01/10530455/1WWdXdJBbTW). 
You can also check out the arXiv version at  [RigidFlowPP-arXiv](https://arxiv.org/abs/2310.11284).

In this repository, we apply RigidFlow++ for self-supervised motion prediction.
For the codes in self-supervised 3D scene flow estimation, please refer to [RigidFlowPP](https://github.com/L1bra1/RigidFlowPP).
The code is created by Ruibo Li (ruibo001@e.ntu.edu.sg).


## Prerequisites
* Python 3.7
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.9.0)


Create a conda environment for RigidPPMotion:
```
conda create -n RigidPPMotion python=3.7
conda activate RigidPPMotion
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=10.2 -c pytorch
pip install numpy tqdm scikit-learn opencv-python matplotlib pyquaternion
```

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).
```bash
cd lib
python setup.py install
cd ../
```

Install & complie supervoxel segmentation method: 
```bash
cd Supervoxel_utils
g++ -std=c++11 -fPIC -shared -o main.so main.cc
cd ../
```
More details about the supervoxel segmentation method, please refer to [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds).

## Data preprocess

### nuScenes
1. Prepare the input data and the motion ground truth:
   - Download the [nuScenes data](https://www.nuscenes.org/), and then follow [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet) to process the training, validation, and test data.
   
2. Prepare raw point clouds and perform ground segmentation for self-supervision:

    - Extract raw point clouds and backward BEV maps for the training samples in nuScenes: 
      ```
      python gen_data/gen_back_BEV_raw_point.py --root /path_to/nuScenes/nuScenes-data/ --split train --savepath /path_to/nuScenes/self-data/
      ```
    - Perform ground segmentation for raw point clouds: 
        ```
      python gen_data/gen_ground_point.py --savepath /path_to/nuScenes/self-data/
      ```
      The data for self-supervised learning will be saved in `/path_to/nuScenes/self-data/`. Please references `gen_data/README.md` for more details.



## Evaluation

### Trained models
The trained model can be downloaded from [model_nuScenes.pth](https://drive.google.com/file/d/1yomzi5vkAJV1howJCxQdOOEm_X0LBp1y/view?usp=drive_link).


### Testing

Run the command:

```
python eval_SelfMotionNet.py --evaldata /path_to/nuScenes/input-data/test/ --pretrained pretrained/model_nuScenes.pth 
```

set `evaldata` to the directory of the test data (e.g., `/path_to/nuScenes/input-data/test/`).

set `pretrained` to the trained model (e.g., `pretrained/model_nuScenes.pth`).



## Training
Run the command:
```
python train_SelfMotionNet.py --motiondata /path_to/nuScenes/input-data/train/ --selfdata /path_to/nuScenes/self-data/ --evaldata /path_to/nuScenes/input-data/val/ 
```

set `motiondata` to the directory of the input training data (e.g., `/path_to/nuScenes/input-data/train/`).

set `selfdata` to the directory of the self-supervised data (e.g., `/path_to/nuScenes/self-data/`).

set `evaldata` to the directory of the input validation or test data (e.g., `/path_to/nuScenes/input-data/val/`).

## Citation

If you find this code useful, please cite our paper:
```
@article{li2024self,
  title={Self-Supervised 3D Scene Flow Estimation and Motion Prediction using Local Rigidity Prior},
  author={Li, Ruibo and Zhang, Chi and Wang, Zhe and Shen, Chunhua and Lin, Guosheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```


## Acknowledgement

Our project references the codes in the following repos.

* [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet)
* [nuScenes](https://github.com/nutonomy/nuscenes-devkit/tree/master)
* [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds)
* [flownet3d_pytorch](https://github.com/hyangwinter/flownet3d_pytorch)
* [WeakMotion](https://github.com/L1bra1/WeakMotion) 
