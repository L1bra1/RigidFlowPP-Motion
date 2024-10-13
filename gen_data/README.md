
## Data preprocess
### nuScenes 
1. Prepare the nuScenes data.

   - Download the [nuScenes data](https://www.nuscenes.org/) and follow [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet) to process the data, we want the data to be saved like this:
   
        ```
        nuScenes
        |-- input-data (the processed input data from MotionNet)
        |-- nuScenes-data (downloaded raw data)
        |   |-- maps
        |   |-- samples
        |   |-- sweeps
        |   |-- v1.0-trainval
        ```
2. Prepare raw point clouds and perform ground segmentation for self-supervision:
    - Extract raw point clouds and backward BEV maps for the training samples in nuScenes: 
      ```
      python gen_data/gen_back_BEV_raw_point.py --root /path_to/nuScenes/nuScenes-data/ --split train --savepath /path_to/nuScenes/self-data/
      ```
    - Perform ground segmentation for raw point clouds: 
        ```
      python gen_data/gen_ground_point.py --savepath /path_to/nuScenes/self-data/
      ```
    
    
The final directory should be like this:

```
nuScenes
|-- input-data (the processed input data from MotionNet)
|-- nuScenes-data (downloaded raw data)
|   |-- maps
|   |-- samples
|   |-- sweeps
|   |-- v1.0-trainval
|
|-- self-data (data for self-supervision)
|   |-- raw-pc
|   |-- sample-info
|   |-- motionnet-data-back 
|   |-- ground-info
```

## Acknowledgement

The data generation references the codes in the following repos.   
* [MotionNet](https://www.merl.com/research/?research=license-request&sw=MotionNet)
* [nuScenes](https://github.com/nutonomy/nuscenes-devkit/tree/master)


