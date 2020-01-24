# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import csv
import os
from math import radians

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from camera_model import CameraModel
from utils import invert_pose, rotate_forward


def get_calib_kitti(sequence):
    if sequence == 0:
        return torch.tensor([718.856, 718.856, 607.1928, 185.2157])
    elif sequence == 3:
        return torch.tensor([721.5377, 721.5377, 609.5593, 172.854])
    elif sequence in [5, 6, 7, 8, 9]:
        return torch.tensor([707.0912, 707.0912, 601.8873, 183.1104])
    else:
        raise TypeError("Sequence Not Available")


class DatasetVisibilityKittiSingle(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, maps_folder='local_maps',
                 use_reflectance=False, max_t=2., max_r=10., split='test', device='cpu', test_sequence='00'):
        super(DatasetVisibilityKittiSingle, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = maps_folder
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}

        self.all_files = []
        self.model = CameraModel()
        self.model.focal_length = [7.18856e+02, 7.18856e+02]
        self.model.principal_point = [6.071928e+02, 1.852157e+02]
        for dir in ['00', '03', '05', '06', '07', '08', '09']:
            self.GTs_R[dir] = []
            self.GTs_T[dir] = []
            df_locations = pd.read_csv(os.path.join(dataset_dir, dir, 'poses.csv'), sep=',', dtype={'timestamp': str})
            for index, row in df_locations.iterrows():
                if not os.path.exists(os.path.join(dataset_dir, dir, maps_folder, str(row['timestamp'])+'.h5')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, dir, 'image_2', str(row['timestamp'])+'.png')):
                    continue
                if dir == test_sequence and split.startswith('test'):
                    self.all_files.append(os.path.join(dir, str(row['timestamp'])))
                elif (not dir == test_sequence) and split == 'train':
                    self.all_files.append(os.path.join(dir, str(row['timestamp'])))
                GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
                GT_T = np.array([row['x'], row['y'], row['z']])
                self.GTs_R[dir].append(GT_R)
                self.GTs_T[dir].append(GT_T)

        self.test_RT = []
        if split == 'test':
            test_RT_file = os.path.join(dataset_dir, f'test_RT_seq{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(test_RT_file):
                print(f'TEST SET: Using this file: {test_RT_file}')
                df_test_RT = pd.read_csv(test_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {test_RT_file}')
                print("Generating a new one")
                test_RT_file = open(test_RT_file, 'w')
                test_RT_file = csv.writer(test_RT_file, delimiter=',')
                test_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    test_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.test_RT.append([i, transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.test_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        run = str(item.split('/')[0])
        timestamp = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, run, 'image_2', timestamp+'.png')
        pc_path = os.path.join(self.root_dir, run, self.maps_folder, timestamp+'.h5')

        try:
            with h5py.File(pc_path, 'r') as hf:
                pc = hf['PC'][:]
                if self.use_reflectance:
                    reflectance = hf['intensity'][:]
                    reflectance = torch.from_numpy(reflectance).float()
        except Exception as e:
            print(f'File Broken: {pc_path}')
            raise e

        pc_in = torch.from_numpy(pc.astype(np.float32))#.float()
        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[1, :] *= -1

        img = Image.open(img_path)
        img_rotation = 0.
        if self.split == 'train':
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split != 'test':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.test_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = get_calib_kitti(int(run))
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        if not self.use_reflectance:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'idx': int(run), 'rgb_name': timestamp}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'reflectance': reflectance, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'idx': int(run), 'rgb_name': timestamp}

        return sample
