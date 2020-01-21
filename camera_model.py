# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import numpy as np
import torch


class CameraModel:

    def __init__(self, focal_length=None, principal_point=None):
        self.focal_length = focal_length
        self.principal_point = principal_point

    def project_pytorch(self, xyz: torch.Tensor, image_size, reflectance=None):
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
                raise TypeError("Wrong Coordinates")
        order = [1, 2, 0, 3]
        xyzw = xyz[order, :]
        indexes = xyzw[2, :] >= 0
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]

        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0]
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        indexes = uv[0, :] >= 0.1
        indexes = indexes & (uv[1, :] >= 0.1)
        indexes = indexes & (uv[0,:] < image_size[1])
        indexes = indexes & (uv[1,:] < image_size[0])
        if reflectance is None:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], None
        else:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], reflectance[:, indexes]

        return uv

    def get_matrix(self):
        matrix = np.zeros([3, 3])
        matrix[0, 0] = self.focal_length[0]
        matrix[1, 1] = self.focal_length[1]
        matrix[0, 2] = self.principal_point[0]
        matrix[1, 2] = self.principal_point[1]
        matrix[2, 2] = 1.0
        return matrix
