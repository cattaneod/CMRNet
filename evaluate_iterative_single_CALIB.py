# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import csv
import random

import cv2
import mathutils
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import visibility
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from skimage import io
from tqdm import tqdm

from camera_model import CameraModel
from DatasetVisibilityKitti import DatasetVisibilityKittiSingle
from models.CMRNet.CMRNet import CMRNet
from quaternion_distances import quaternion_distance
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

ex = Experiment("CMRNet-evaluate-iterative")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'kitti'
    data_folder = './KITTI/sequences/'
    test_sequence = 0
    use_prev_output = False
    max_t = 2.
    max_r = 10.
    occlusion_kernel = 5
    occlusion_threshold = 3.0
    network = 'PWC_f1'
    norm = 'bn'
    show = False
    use_reflectance = False
    weight = None  # List of weights' path, for iterative refinement
    save_name = None
    # Set to True only if you use two network, the first for rotation and the second for translation
    rot_transl_separated = False
    random_initial_pose = False
    save_log = False
    maps_folder = None


weights = [
    '/checkpoints/kitti/iterative_final_test/kitti_iter1.tar',
    '/checkpoints/kitti/iterative_final_test/kitti_iter2.tar',
    '/checkpoints/kitti/iterative_final_test/kitti_iter3.tar',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@ex.automain
def main(_config, seed):
    global EPOCH, weights
    if _config['weight'] is not None:
        weights = _config['weight']

    dataset_class = DatasetVisibilityKittiSingle
    img_shape = (384, 1280)

    split = 'test'
    if _config['random_initial_pose']:
        split = 'test_random'
    maps_folder = 'local_maps'
    if _config['maps_folder'] is not None:
        maps_folder = _config['maps_folder']

    if _config['test_sequence'] is None:
        raise TypeError('test_sequences cannot be None')
    else:
        if isinstance(_config['test_sequence'], int):
            _config['test_sequence'] = f"{_config['test_sequence']:02d}"
        dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                    split=split, use_reflectance=_config['use_reflectance'], maps_folder=maps_folder,
                                    test_sequence=_config['test_sequence'])

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x):
        return _init_fn(x, seed)

    num_worker = 6
    batch_size = 1

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    print(len(TestImgLoader))

    models = []
    for i in range(len(weights)):
        if _config['network'].startswith('PWC'):
            feat = 1
            md = 4
            split = _config['network'].split('_')
            for item in split[1:]:
                if item.startswith('f'):
                    feat = int(item[-1])
                elif item.startswith('md'):
                    md = int(item[2:])
            assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
            assert 0 < md, "md must be positive"
            model = CMRNet(img_shape, use_feat_from=feat, md=md,
                           use_reflectance=_config['use_reflectance'])
        else:
            raise TypeError("Network unknown")
        checkpoint = torch.load(weights[i], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
        if i == 0:
            _config['occlusion_threshold'] = checkpoint['config']['occlusion_threshold']
            _config['occlusion_kernel'] = checkpoint['config']['occlusion_kernel']
        else:
            assert _config['occlusion_threshold'] == checkpoint['config']['occlusion_threshold']
            assert _config['occlusion_kernel'] == checkpoint['config']['occlusion_kernel']

    if _config['save_log']:
        log_file = f'./results_for_paper/log_seq{_config["test_sequence"]}.csv'
        log_file = open(log_file, 'w')
        log_file = csv.writer(log_file)
        header = ['frame']
        for i in range(len(weights) + 1):
            header += [f'iter{i}_error_t', f'iter{i}_error_r', f'iter{i}_error_x', f'iter{i}_error_y',
                       f'iter{i}_error_z', f'iter{i}_error_r', f'iter{i}_error_p', f'iter{i}_error_y']
        log_file.writerow(header)

    show = _config['show']

    errors_r = []
    errors_t = []
    errors_t2 = []
    errors_rpy = []
    all_RTs = []

    prev_tr_error = None
    prev_rot_error = None

    for i in range(len(weights) + 1):
        errors_r.append([])
        errors_t.append([])
        errors_t2.append([])
        errors_rpy.append([])

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):

        log_string = [str(batch_idx)]

        lidar_input = []
        rgb_input = []
        shape_pad = [0, 0, 0, 0]

        if batch_idx == 0 or not _config['use_prev_output']:
            # Qui dare posizione di input del frame corrente rispetto alla GT
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
        else:
            sample['tr_error'] = prev_tr_error
            sample['rot_error'] = prev_rot_error

        for idx in range(len(sample['rgb'])):

            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            # ProjectPointCloud in RT-pose
            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            pc_rotated = sample['point_cloud'][idx].clone()
            reflectance = None
            if _config['use_reflectance']:
                reflectance = sample['reflectance'][idx].cuda()

            R = mathutils.Quaternion(sample['rot_error'][idx])
            T = mathutils.Vector(sample['tr_error'][idx])

            pc_rotated = rotate_back(pc_rotated, R, T)
            cam_params = sample['calib'][idx].cuda()
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            uv, depth, points, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance)
            uv = uv.t().int()
            depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            depth_img += 1000.
            depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
            depth_img[depth_img == 1000.] = 0.

            projected_points = torch.zeros_like(depth_img, device='cuda')
            projected_points = visibility.visibility2(depth_img, cam_params, projected_points, depth_img.shape[1],
                                                      depth_img.shape[0], _config['occlusion_threshold'],
                                                      _config['occlusion_kernel'])

            if _config['use_reflectance']:
                uv = uv.long()
                indexes = projected_points[uv[:, 1], uv[:, 0]] == depth
                refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                refl_img[uv[indexes, 1], uv[indexes, 0]] = refl[0, indexes]

            projected_points /= 100.
            if not _config['use_reflectance']:
                projected_points = projected_points.unsqueeze(0)
            else:
                projected_points = torch.stack((projected_points, refl_img))

            rgb = sample['rgb'][idx].cuda()

            shape_pad[3] = (img_shape[0] - rgb.shape[1])
            shape_pad[1] = (img_shape[1] - rgb.shape[2])

            rgb = F.pad(rgb, shape_pad)
            projected_points = F.pad(projected_points, shape_pad)

            rgb_input.append(rgb)
            lidar_input.append(projected_points)

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)
        if show:
            out0 = overlay_imgs(rgb, lidar_input)

            cv2.imshow("INPUT", out0[:, :, [2, 1, 0]])
            cv2.waitKey(1)

            pc_GT = sample['point_cloud'][idx].clone()

            uv, depth, _, refl = cam_model.project_pytorch(pc_GT, real_shape)
            uv = uv.t().int()
            depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            depth_img += 1000.
            depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
            depth_img[depth_img == 1000.] = 0.

            projected_points = torch.zeros_like(depth_img, device='cuda')
            projected_points = visibility.visibility2(depth_img, cam_params, projected_points, depth_img.shape[1],
                                                      depth_img.shape[0], _config['occlusion_threshold'],
                                                      _config['occlusion_kernel'])
            projected_points /= 100.

            projected_points = F.pad(projected_points, shape_pad)

            lidar_GT = projected_points.unsqueeze(0).unsqueeze(0)
            out1 = overlay_imgs(rgb_input[0], lidar_GT)
            # cv2.imshow("GT", out1[:, :, [2, 1, 0]])
            # plt.figure()
            # plt.imshow(out1)
            # if batch_idx == 0:
            # out2 = overlay_imgs(sample['rgb'][0], lidar_input)
            # plt.figure()
            # plt.imshow(out2)
            # io.imshow(lidar_input[0][0].cpu().numpy(), cmap='jet')
            # io.show()
        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)
        target_transl = sample['tr_error'].to(device)
        target_rot = sample['rot_error'].to(device)

        point_cloud = sample['point_cloud'][0].to(device)
        reflectance = None
        if _config['use_reflectance']:
            reflectance = sample['reflectance'][0].to(device)
        camera_model = cam_model

        R = quat2mat(target_rot[0])
        T = tvector2mat(target_transl[0])
        RT1_inv = torch.mm(T, R)
        RT1 = RT1_inv.clone().inverse()

        rotated_point_cloud = rotate_forward(point_cloud, RT1)
        RTs = [RT1]

        T_composed = RT1[:3, 3]
        R_composed = quaternion_from_matrix(RT1)
        errors_t[0].append(T_composed.norm().item())
        errors_t2[0].append(T_composed)
        errors_r[0].append(quaternion_distance(R_composed.unsqueeze(0),
                                               torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                               R_composed.device))
        # rpy_error = quaternion_to_tait_bryan(R_composed)
        rpy_error = mat2xyzrpy(RT1)[3:]

        rpy_error *= (180.0 / 3.141592)
        errors_rpy[0].append(rpy_error)
        log_string += [str(errors_t[0][-1]), str(errors_r[0][-1]), str(errors_t2[0][-1][0].item()),
                       str(errors_t2[0][-1][1].item()), str(errors_t2[0][-1][2].item()),
                       str(errors_rpy[0][-1][0].item()), str(errors_rpy[0][-1][1].item()),
                       str(errors_rpy[0][-1][2].item())]

        if batch_idx == 0.:
            print(f'Initial T_erorr: {errors_t[0]}')
            print(f'Initial R_erorr: {errors_r[0]}')
        start = 0

        # Run model
        with torch.no_grad():
            for iteration in range(start, len(weights)):
                # Run the i-th network
                T_predicted, R_predicted = models[iteration](rgb, lidar)
                if _config['rot_transl_separated'] and iteration == 0:
                    T_predicted = torch.tensor([[0., 0., 0.]], device='cuda')
                if _config['rot_transl_separated'] and iteration == 1:
                    R_predicted = torch.tensor([[1., 0., 0., 0.]], device='cuda')

                # Project the points in the new pose predicted by the i-th network
                R_predicted = quat2mat(R_predicted[0])
                T_predicted = tvector2mat(T_predicted[0])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                RTs.append(torch.mm(RTs[iteration], RT_predicted))

                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                uv2, depth2, _, refl = camera_model.project_pytorch(rotated_point_cloud, real_shape, reflectance)
                uv2 = uv2.t().int()
                depth_img2 = torch.zeros(real_shape[:2], device=device)
                depth_img2 += 1000.
                depth_img2 = visibility.depth_image(uv2, depth2, depth_img2, uv2.shape[0], real_shape[1], real_shape[0])
                depth_img2[depth_img2 == 1000.] = 0.

                out_cuda2 = torch.zeros_like(depth_img2, device=device)
                out_cuda2 = visibility.visibility2(depth_img2, cam_params,
                                                   out_cuda2, depth_img2.shape[1],
                                                   depth_img2.shape[0], _config['occlusion_threshold'],
                                                   _config['occlusion_kernel'])

                if _config['use_reflectance']:
                    uv = uv.long()
                    indexes = projected_points[uv[:, 1], uv[:, 0]] == depth
                    refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    refl_img[uv[indexes, 1], uv[indexes, 0]] = refl[0, indexes]
                    refl_img = F.pad(refl_img, shape_pad)

                out_cuda2 = F.pad(out_cuda2, shape_pad)

                lidar = out_cuda2.clone()
                lidar /= 100.
                if not _config['use_reflectance']:
                    lidar = lidar.unsqueeze(0)
                else:
                    lidar = torch.stack((lidar, refl_img))
                lidar = lidar.unsqueeze(0)
                if show:
                    out3 = overlay_imgs(rgb[0], lidar, idx=batch_idx)
                    cv2.imshow(f'Iter_{iteration}', out3[:, :, [2, 1, 0]])
                    cv2.waitKey(1)
                    # if iter == 1:
                    # plt.figure()
                    # plt.imshow(out3)
                    # io.imshow(lidar.cpu().numpy(), cmap='jet')
                    # io.show()

                T_composed = RTs[iteration + 1][:3, 3]
                R_composed = quaternion_from_matrix(RTs[iteration + 1])
                errors_t[iteration + 1].append(T_composed.norm().item())
                errors_t2[iteration + 1].append(T_composed)
                errors_r[iteration + 1].append(quaternion_distance(R_composed.unsqueeze(0),
                                                                   torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                                                   R_composed.device))

                # rpy_error = quaternion_to_tait_bryan(R_composed)
                rpy_error = mat2xyzrpy(RTs[iteration + 1])[3:]
                rpy_error *= (180.0 / 3.141592)
                errors_rpy[iteration + 1].append(rpy_error)
                log_string += [str(errors_t[iteration + 1][-1]), str(errors_r[iteration + 1][-1]),
                               str(errors_t2[iteration + 1][-1][0].item()), str(errors_t2[iteration + 1][-1][1].item()),
                               str(errors_t2[iteration + 1][-1][2].item()), str(errors_rpy[iteration + 1][-1][0].item()),
                               str(errors_rpy[iteration + 1][-1][1].item()), str(errors_rpy[iteration + 1][-1][2].item())]


        all_RTs.append(RTs[-1])
        prev_RT = RTs[-1].inverse()
        prev_tr_error = prev_RT[:3, 3].unsqueeze(0)
        prev_rot_error = quaternion_from_matrix(prev_RT).unsqueeze(0)
        # Qui prev_rt è quanto si discosta l'output della rete rispetto alla GT

        if _config['save_log']:
            log_file.writerow(log_string)

    if _config['save_log']:
        log_file.close()
    print("Iterative refinement: ")
    for i in range(len(weights) + 1):
        errors_r[i] = torch.tensor(errors_r[i]) * (180.0 / 3.141592)
        errors_t[i] = torch.tensor(errors_t[i]) * 100
        print(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
              f"     Mean Rotation Error: {errors_r[i].mean():.4f} °")
        print(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
              f"     Median Rotation Error: {errors_r[i].median():.4f} °\n")

    print("-------------------------------------------------------")
    print("Timings:")
    for i in range(len(errors_t2)):
        errors_t2[i] = torch.stack(errors_t2[i])
        errors_rpy[i] = torch.stack(errors_rpy[i])
    plt.plot(errors_t2[-1][:, 0].cpu().numpy())
    plt.show()
    plt.plot(errors_t2[-1][:, 1].cpu().numpy())
    plt.show()
    plt.plot(errors_t2[-1][:, 2].cpu().numpy())
    plt.show()

    if _config["save_name"] is not None:
        torch.save(torch.stack(errors_t).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_t')
        torch.save(torch.stack(errors_r).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_r')
        torch.save(torch.stack(errors_t2).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_t2')
        torch.save(torch.stack(errors_rpy).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_rpy')

    print("End!")
