import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
import matplotlib.pyplot as plt


class SDM(nn.Module):
    def __init__(self, img_shape, train_num, point_shape_factor=1, device='cuda'):
        super().__init__()
        self.device = device
        # self.space = space  # 多少个像素取一个控制点
        self.img_shape = img_shape
        point_shape = [i // point_shape_factor for i in img_shape]
        self.point_shape = point_shape
        self.point_shape_factor = torch.tensor((np.array(self.img_shape) - 1) / (np.array(self.point_shape) - 1))

        self.train_num = train_num

        # 控制点原位置，[3, mx, my, mz]
        self.control_points_origin_position = (self.point_shape_factor.view(3, 1, 1, 1) * self.generate_tensor(
            self.point_shape)).float().to(self.device)
        # 控制点相对原位置移动位置，[n, 3, mx, my, mz]
        self.control_points = nn.Parameter(
            torch.zeros([self.train_num, 3, self.point_shape[0], self.point_shape[1], self.point_shape[2]]),
            requires_grad=True)

        # 使用四元数表示旋转
        self.quaternion = nn.Parameter(torch.Tensor([1, 0, 0, 0]).repeat([self.train_num, 1]))
        # 平移
        self.bias = nn.Parameter(torch.Tensor([0, 0, 0]).repeat([self.train_num, 1]))

    def forward(self, image_from, idx):
        idx = int(idx)
        # plt.figure()
        # plt.imshow(image_from[:, :, 40].cpu().numpy(), cmap='gray')
        # plt.show()

        # 图像对应控制点 (1, 3, x, y, z)
        control_points = self.control_points[[idx]] + self.control_points_origin_position[None]
        # (1, x, y, z, 3)
        grid = self.generate_tensor(self.img_shape, normalize=True).to(self.device).permute((1, 2, 3, 0))[None]
        # 记录图像个点变换后在原图的位置
        transformed_point = F.grid_sample(control_points, grid, padding_mode="border", align_corners=True)
        # 进行全局的旋转平移
        rotation_matrix = self.as_rotation_matrix(self.quaternion[idx])
        # (1, x, y, z, 3)
        transformed_point = transformed_point.permute(0, 2, 3, 4, 1) @ rotation_matrix + self.bias[idx]
        for i in range(3):
            transformed_point[:, :, :, :, i] = 2 * transformed_point[:, :, :, :, i] / (self.img_shape[i] - 1) - 1

        image_to = F.grid_sample(image_from[None, None], transformed_point.double(), padding_mode="border",
                                 align_corners=True)[0, 0]

        # plt.figure()
        # plt.imshow(image_to[:, :, 40].cpu().detach().numpy(), cmap='gray')
        # plt.show()

        return image_to, rotation_matrix

    @staticmethod
    def generate_tensor(shape, normalize=False):
        x, y, z = shape
        # 生成坐标网格
        grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z), indexing='ij')
        # 将坐标合并成一个张量
        if normalize:
            coordinates = torch.stack((2 * grid_x / (x - 1) - 1,
                                       2 * grid_y / (y - 1) - 1,
                                       2 * grid_z / (z - 1) - 1), dim=0)
        else:
            coordinates = torch.stack((grid_x, grid_y, grid_z), dim=0)

        return coordinates

    def as_rotation_matrix(self, quaternion):
        rt_mat = torch.zeros([9]).to(self.device)

        rt_mat[0] = 1 - 2 * (quaternion[2] * quaternion[2]) - 2 * (quaternion[3] * quaternion[3])
        rt_mat[1] = 2 * quaternion[1] * quaternion[2] - 2 * quaternion[0] * quaternion[3]
        rt_mat[2] = 2 * quaternion[1] * quaternion[3] + 2 * quaternion[0] * quaternion[2]
        rt_mat[3] = 2 * quaternion[1] * quaternion[2] + 2 * quaternion[0] * quaternion[3]
        rt_mat[4] = 1 - 2 * (quaternion[1] * quaternion[1]) - 2 * (quaternion[3] * quaternion[3])
        rt_mat[5] = 2 * quaternion[2] * quaternion[3] - 2 * quaternion[0] * quaternion[1]
        rt_mat[6] = 2 * quaternion[1] * quaternion[3] - 2 * quaternion[0] * quaternion[2]
        rt_mat[7] = 2 * quaternion[2] * quaternion[3] + 2 * quaternion[0] * quaternion[1]
        rt_mat[8] = 1 - 2 * (quaternion[1] * quaternion[1]) - 2 * (quaternion[2] * quaternion[2])

        return rt_mat.reshape([3, 3]).permute(1, 0)
