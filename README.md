[![CC BY-NC-SA 4.0][cc-by-sa-shield]][cc-by-sa]

## CMRNet: Camera to LiDAR-Map Registration (IEEE ITSC 2019)

### License
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-sa].
[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

### News

##### Check out our new paper "CMRNet++: Map and Camera Agnostic Monocular Visual Localization in LiDAR Maps":
* [PDF](https://arxiv.org/abs/2004.13795)
* [Demo](http://rl.uni-freiburg.de/research/vloc-in-lidar)
* [Video](https://www.youtube.com/watch?v=EUCloC6flr4)

[<img src="video-preview.png" width="512">](https://www.youtube.com/watch?v=EUCloC6flr4) 

---
> #### 2020/06/24
> * We released the pretrained weights, see [Pretrained Model](#pretrained-model).
> 
> #### 2020/05/11
> * We released the SLAM ground truth files, see [Local Maps Generation](#local-maps-generation).
> * Multi-GPU training.
> * Added requirements.txt


### Code

![CMRNet Teaser](./teaser.png)

PyTorch implementation of CMRNet.

This code is a WIP (Work In Progress), use at your own risk.
This version only works on GPUs (no CPU version available).

Tested on:
* Ubuntu 16.04
* python 3.6
* cuda 9.0
* pytorch 1.0.1.post2

Dependencies (this list is not complete):
* [sacred](https://sacred.readthedocs.io/)
* mathutils (use this version: https://gitlab.com/m1lhaus/blender-mathutils)
* openCV (for visualization)
* open3d 0.7 (only for maps preprocessing)
* [pykitti](https://github.com/utiasSTARS/pykitti) (only for maps preprocessing)

### Installation
Install the required packages:
```
pip install -r requirements.txt
```

It is recommended to use a dedicated conda environment
```
conda create -n 'cmrnet' python=3.6
conda activate cmrnet
pip install -r requirements.txt
```

### Data

We trained and tested CMRNet on the [KITTI odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) sequences 00, 03, 05, 06, 07, 08, and 09.

We used a LiDAR-based SLAM system to generate the ground truths.

The Data Loader requires a local point cloud for each camera frame, the point cloud must be expressed with respect to the camera_2 reference frame, BUT (very important) with a different axes representation: X-forward, Y-right, Z-down.

For reading speed and file size we decided to save the point clouds as h5 files.

The directory structure should looks like:
```bash
KITTI_ODOMETRY
├── 00
│   ├── image_2
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   ├── ...
│   │   └── 004540.png
│   ├── local_maps
│   │   ├── 000000.h5
│   │   ├── 000001.h5
│   │   ├── ...
│   │   └── 004540.h5
│   └── poses.csv
└── 03
    ├── image_2
    │   ├── 000000.png
    │   ├── 000001.png
    │   ├── ...
    │   └── 000800.png
    ├── local_maps
    │   ├── 000000.h5
    │   ├── 000001.h5
    │   ├── ...
    │   └── 000800.h5
    └── poses.csv

```

#### Local Maps Generation

To generate the h5 files, use the script preprocess/kitti_maps.py with the ground truth files in data/.

In the sequence 08, the SLAM failed to detect a loop closure, so the poses are not coherent around that closure.
Therefore, we splitted the map at frame 3000, so to have two coherent maps for that sequence.

```bash
python preprocess/kitti_maps.py --sequence 00 --kitti_folder ./KITTI_ODOMETRY/
python preprocess/kitti_maps.py --sequence 03 --kitti_folder ./KITTI_ODOMETRY/
python preprocess/kitti_maps.py --sequence 05 --kitti_folder ./KITTI_ODOMETRY/
python preprocess/kitti_maps.py --sequence 06 --kitti_folder ./KITTI_ODOMETRY/
python preprocess/kitti_maps.py --sequence 07 --kitti_folder ./KITTI_ODOMETRY/
python preprocess/kitti_maps.py --sequence 08 --kitti_folder ./KITTI_ODOMETRY/ --end 3000
python preprocess/kitti_maps.py --sequence 08 --kitti_folder ./KITTI_ODOMETRY/ --start 3000
python preprocess/kitti_maps.py --sequence 09 --kitti_folder ./KITTI_ODOMETRY/
```

### Single Iteration example

Training:
```bash
python main_visibility_CALIB.py with batch_size=24 data_folder=./KITTI_ODOMETRY/ epochs=300 max_r=10 max_t=2 BASE_LEARNING_RATE=0.0001 savemodel=./checkpoints/ test_sequence=0
```

Evaluation:
```bash
python evaluate_iterative_single_CALIB.py with test_sequence=00 maps_folder=local_maps data_folder=./KITTI_ODOMETRY/ weight="['./checkpoints/weights.tar']"
```

### Iterative refinement example

Training
```bash
python main_visibility_CALIB.py with batch_size=24 data_folder=./KITTI_ODOMETRY/ epochs=300 max_r=10 max_t=2   BASE_LEARNING_RATE=0.0001 savemodel=./checkpoints/ test_sequence=0
python main_visibility_CALIB.py with batch_size=24 data_folder=./KITTI_ODOMETRY/ epochs=300 max_r=2  max_t=1   BASE_LEARNING_RATE=0.0001 savemodel=./checkpoints/ test_sequence=0
python main_visibility_CALIB.py with batch_size=24 data_folder=./KITTI_ODOMETRY/ epochs=300 max_r=2  max_t=0.6 BASE_LEARNING_RATE=0.0001 savemodel=./checkpoints/ test_sequence=0
```

Evaluation
```bash
python evaluate_iterative_single_CALIB.py with test_sequence=00 maps_folder=local_maps data_folder=./KITTI_ODOMETRY/ weight="['./checkpoints/iter1.tar','./checkpoints/iter2.tar','./checkpoints/iter3.tar']"
```

### Pretrained Model
The weights for the three iterations, trained on the sequences 03, 05, 06, 07, 08 and 09 are available here:
[Iteration 1](https://drive.google.com/file/d/1cwUGSrlVrD_WzKpTPgsUQNcdaCoh0aeE)
[Iteration 2](https://drive.google.com/file/d/1ffMFrspQoGaIY5YJsy0LMEzLqrd5XCnb)
[Iteration 3](https://drive.google.com/file/d/1-SYluv-hVDA6gebvCJrXtKuxCYdHqebw)

Results:
|| <p>Median <br> Transl. error</p> | <p>Median <br> Rotation. error</p> |
|---|---|---|
| Iteration 1 | 0.46 cm | 1.60° |
| Iteration 2 | 0.25 cm | 1.14° |
| Iteration 3 | 0.20 cm | 0.97° |

### Paper
"CMRNet: Camera to LiDAR-Map Registration"
* [IEEEXplore](https://ieeexplore.ieee.org/document/8917470)
* [arXiv](https://arxiv.org/abs/1906.10109)
* [Video](https://www.youtube.com/watch?v=ZFI_1HCo_J8)

If you use CMRNet, please cite:
```
@InProceedings{cattaneo2019cmrnet,
  author={Cattaneo, Daniele and Vaghi, Matteo and Ballardini, Augusto Luis and Fontana, Simone and Sorrenti, Domenico Giorgio and Burgard, Wolfram},
  booktitle={2019 IEEE Intelligent Transportation Systems Conference (ITSC)},
  title={CMRNet: Camera to LiDAR-Map Registration},
  year={2019},
  pages={1283-1289},
  doi={10.1109/ITSC.2019.8917470},
  month={Oct}
}
```

If you use the ground truths, please also cite:
```
@INPROCEEDINGS{Caselitz_2016, 
  author={T. {Caselitz} and B. {Steder} and M. {Ruhnke} and W. {Burgard}}, 
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Monocular camera localization in 3D LiDAR maps}, 
  year={2016},
  pages={1926-1931},
  doi={10.1109/IROS.2016.7759304}
}
```

### Acknowledgments
[correlation_package](models/CMRNet/correlation_package) was taken from [flownet2](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)

[PWCNet.py](model/PWC/PWCNet.py) is a modified version of the original [PWC-DC network](https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py)

### Contacts
Daniele Cattaneo (cattaneo@informatik.uni-freiburg.de or d.cattaneo10@campus.unimib.it)


[cc-by-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey
