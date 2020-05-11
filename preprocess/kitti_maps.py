# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")

import h5py
import numpy as np
import open3d as o3
import pykitti
import torch
from tqdm import tqdm

from utils import to_rotation_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--sequence', default='00',
                    help='sequence')
parser.add_argument('--device', default='cuda',
                    help='device')
parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
parser.add_argument('--start', default=0, help='Starting Frame')
parser.add_argument('--end', default=100000, help='End Frame')
parser.add_argument('--map', default=None, help='Use map file')
parser.add_argument('--kitti_folder', default='./KITTI/ODOMETRY', help='Folder of the KITTI dataset')

args = parser.parse_args()
sequence = args.sequence
print("Sequnce: ", sequence)
velodyne_folder = os.path.join(args.kitti_folder, 'sequences', sequence, 'velodyne')
pose_file = os.path.join('./data', f'kitti-{sequence}.csv')

poses = []
with open(pose_file, 'r') as f:
    for x in f:
        if x.startswith('timestamp'):
            continue
        x = x.split(',')
        T = torch.tensor([float(x[1]), float(x[2]), float(x[3])])
        R = torch.tensor([float(x[7]), float(x[4]), float(x[5]), float(x[6])])
        poses.append(to_rotation_matrix(R, T))

map_file = args.map
first_frame = int(args.start)
last_frame = min(len(poses), int(args.end))
kitti = pykitti.odometry(args.kitti_folder, sequence)

if map_file is None:

    pc_map = []
    pcl = o3.PointCloud()
    for i in tqdm(range(first_frame, last_frame)):
        pc = kitti.get_velo(i)
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        intensity = pc[:, 3].copy()
        pc[:, 3] = 1.
        RT = poses[i].numpy()
        pc_rot = np.matmul(RT, pc.T)
        pc_rot = pc_rot.astype(np.float).T.copy()

        pcl_local = o3.PointCloud()
        pcl_local.points = o3.Vector3dVector(pc_rot[:, :3])
        pcl_local.colors = o3.Vector3dVector(np.vstack((intensity, intensity, intensity)).T)

        downpcd = o3.voxel_down_sample(pcl_local, voxel_size=args.voxel_size)

        pcl.points.extend(downpcd.points)
        pcl.colors.extend(downpcd.colors)


    downpcd_full = o3.voxel_down_sample(pcl, voxel_size=args.voxel_size)
    downpcd, ind = o3.statistical_outlier_removal(downpcd_full, nb_neighbors=40, std_ratio=0.3)
    #o3.draw_geometries(downpcd)
    o3.write_point_cloud(f'./map-{sequence}_{args.voxel_size}_{first_frame}-{last_frame}.pcd', downpcd)
else:
    downpcd = o3.read_point_cloud(map_file)


voxelized = torch.tensor(downpcd.points, dtype=torch.float)
voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
voxelized = voxelized.t()
voxelized = voxelized.to(args.device)
vox_intensity = torch.tensor(downpcd.colors, dtype=torch.float)[:, 0:1].t()

velo2cam2 = torch.from_numpy(kitti.calib.T_cam2_velo).float().to(args.device)

# SAVE SINGLE PCs
if not os.path.exists(os.path.join(args.kitti_folder, 'sequences', sequence,
                                   f'local_maps_{args.voxel_size}')):
    os.mkdir(os.path.join(args.kitti_folder, 'sequences', sequence, f'local_maps_{args.voxel_size}'))
for i in tqdm(range(first_frame, last_frame)):
    pose = poses[i]
    pose = pose.to(args.device)
    pose = pose.inverse()

    local_map = voxelized.clone()
    local_intensity = vox_intensity.clone()
    local_map = torch.mm(pose, local_map).t()
    indexes = local_map[:, 1] > -25.
    indexes = indexes & (local_map[:, 1] < 25.)
    indexes = indexes & (local_map[:, 0] > -10.)
    indexes = indexes & (local_map[:, 0] < 100.)
    local_map = local_map[indexes]
    local_intensity = local_intensity[:, indexes]

    local_map = torch.mm(velo2cam2, local_map.t())
    local_map = local_map[[2, 0, 1, 3], :]

    #pcd = o3.PointCloud()
    #pcd.points = o3.Vector3dVector(local_map[:,:3].numpy())
    #o3.write_point_cloud(f'{i:06d}.pcd', pcd)

    file = os.path.join(args.kitti_folder, 'sequences', sequence,
                        f'local_maps_{args.voxel_size}', f'{i:06d}.h5')
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)
        hf.create_dataset('intensity', data=local_intensity.cpu().half(), compression='lzf', shuffle=True)
