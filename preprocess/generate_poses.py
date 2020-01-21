# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import csv
import os

dataset_folder = './KITTI/ODOMETRY/sequences'
gt_folder = './improved-gt'
sequences = ['00', '03', '04', '05', '06', '07', '08', '09']
for sequence in sequences:
    print(sequence)
    gt_file = os.path.join(gt_folder, f'kitti-{sequence}.gm2dl')
    out_file = os.path.join(dataset_folder, sequence, 'poses.csv')

    gt_file = open(gt_file, 'r')
    out_file = open(out_file, 'w')
    out_csv = csv.writer(out_file, delimiter=',')
    out_csv.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    for line in gt_file:
        if line.startswith('VERTEX_SE3'):
            line = line.split(' ')
            line = line[1:-1]
            line[0] = f'{int(line[0]):06d}'
            out_csv.writerow(line)

    gt_file.close()
    out_file.close()
