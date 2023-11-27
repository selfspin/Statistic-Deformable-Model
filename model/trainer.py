import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pytz
from tqdm import tqdm
import imageio
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.decomposition import PCA
from model.image import ImgModel


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


class PSNRCriterion(nn.Module):
    def __init__(self, max_pixel_value=1.0):
        super(PSNRCriterion, self).__init__()
        self.max_pixel_value = max_pixel_value

    def forward(self, output, target):
        mse = nn.functional.mse_loss(output, target)
        psnr = 20 * torch.log10(self.max_pixel_value / torch.sqrt(mse))
        return -psnr  # 返回 -PSNR，因为我们要最小化损失


class Trainer:
    def __init__(self, model, train_dataset, batch_size=1, learning_rate=0.05, root=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        self.optimizer = optim.Adam([{'params': [self.model.quaternion], 'lr': 1e-2 * learning_rate},
                                     {'params': [self.model.control_points], 'lr': learning_rate},
                                     {'params': [self.model.bias], 'lr': 1e-2 * learning_rate}])
        self.scheduler = None

        # 配准loss的权重
        # rotation_matrix, control_points, bias
        self.lam = torch.tensor([0, 5e-5, 0]).to(self.device)

        # 将时间戳转换为日期和时间
        local_time = datetime.now()

        # 设置时区为北京时间
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = local_time.astimezone(beijing_tz)
        if root is None:
            self.root = 'outputs/FFD/' + beijing_time.strftime('%Y-%m-%d %H:%M:%S')
            os.makedirs(self.root)
        else:
            self.root = root

        self.pca_model = None
        self.pca_mean = None
        self.pca_features = None
        self.train_pca_cord = None

        self.image_model = None

    def criterion(self, transformed_img, refer_img, rotation_matrix, idx):
        # img_criterion = PSNRCriterion()
        img_criterion = nn.L1Loss()

        loss = img_criterion(transformed_img, refer_img)
        if self.lam[0] > 0:
            loss += self.lam[0] * torch.norm(rotation_matrix - torch.eye(3).to(self.device))

        for name, para in self.model.named_parameters():
            # print(name)
            if self.lam[1] > 0 and 'control_points' in name:
                loss += self.lam[1] * para[idx].norm(p=2, dim=1).mean()
            elif self.lam[2] > 0 and 'bias' in name:
                loss += self.lam[2] * para[idx].norm(p=2)

        return loss

    def train_iter(self, i, iter, idx, refer_img, train_img):
        self.model.train()
        total_loss = 0.0

        self.optimizer.zero_grad()
        transformed_img, rotation_matrix = self.model(train_img, idx)
        loss = self.criterion(transformed_img, refer_img, rotation_matrix, idx)
        loss.backward()
        self.optimizer.step()
        #
        total_loss += loss.item()

        if iter == 0 or (iter + 1) % 50 == 0:
            img = torch.cat([transformed_img[:, :, 40], refer_img[:, :, 40]], dim=1).cpu().detach().numpy()
            plt.imsave(self.root + f"/{i:02d}/{iter + 1:04d}.jpg", img, cmap='gray', vmin=0, vmax=1)

        return total_loss

    def train(self, num_iters=200):
        self.model.train()
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=0.05)
        for i, batch in enumerate(self.train_loader):
            print(f'Generate control points of {i}')
            os.makedirs(self.root + f'/{i:02d}')

            idx, refer_img, train_img = batch['idx'], batch['refer_img'][0].to(self.device), batch['train_img'][0].to(
                self.device)

            iters = tqdm(range(num_iters))
            for iter in iters:
                train_loss = self.train_iter(i, iter, idx, refer_img, train_img)
                iters.set_postfix({'loss': train_loss, 'lr': '%.4f' % self.optimizer.param_groups[0]['lr']})

                # self.scheduler.step()

        print('Training finished.')
        os.makedirs(self.root + '/ckpt')
        torch.save(self.model.state_dict, self.root + '/ckpt/' + 'SDM_weights.pth')
        print('Model saved.')

    def eval(self):
        self.model.eval()
        for i, batch in enumerate(self.train_loader):
            print(f'Eval control points on {i}')
            # os.makedirs(self.root + f'/{i:02d}')

            idx, refer_img, train_img = batch['idx'], batch['refer_img'][0].to(self.device), batch['train_img'][0].to(
                self.device)
            transformed_img, _ = self.model(train_img, idx)

            video_writer = imageio.get_writer(os.path.join(self.root, f'{i:02d}/video.mp4'), fps=20)
            for z in range(transformed_img.shape[2]):
                img = torch.cat([train_img[:, :, z], transformed_img[:, :, z], refer_img[:, :, z]],
                                dim=1).cpu().detach().numpy()
                img = to8b(img)
                # plt.imsave(self.root + f"/{i:02d}/{iter + 1:04d}.jpg", img, cmap='gray', vmin=0, vmax=1)
                video_writer.append_data(img)

            # 关闭视频写入器
            video_writer.close()
            print('Done')

    def apply_pca(self, n_components):
        self.model.eval()
        print('Building PCA model')

        control_points = self.model.control_points.data
        origin_shape = control_points.shape
        train_num = control_points.shape[0]
        control_points = control_points.reshape([train_num, -1])
        assert torch.equal(control_points.reshape(origin_shape), self.model.control_points.data)
        control_points = control_points.cpu()

        self.pca_model = PCA(n_components=n_components)
        data_pca = self.pca_model.fit_transform(control_points)

        print("Original data shape:", control_points.shape)
        print("PCA transformed data shape:", data_pca.shape)
        print("Explained variance:", self.pca_model.explained_variance_ratio_.sum())

        self.pca_mean = torch.from_numpy(self.pca_model.mean_).to(self.device)
        self.pca_features = torch.from_numpy(self.pca_model.components_).to(self.device)
        self.train_pca_cord = torch.from_numpy(data_pca).to(self.device)

        print('PCA model Built')

        # x = self.train_pca_cord @ self.pca_features + self.pca_mean
        # print(control_points.norm())
        # print((x.cpu() - control_points).norm())

        self.image_model = ImgModel(img_shape=(96, 80, 96), point_shape_factor=2, pca_mean=self.pca_mean,
                                    pca_features=self.pca_features).to(self.device)

        return

    def analyse_test_image(self, num_iters=2000):
        if self.image_model is None:
            print('PCA is not applied!')
            return

        os.makedirs(os.path.join(self.root, 'test'), exist_ok=True)
        self.image_model.train()

        data_iter = iter(self.train_loader)
        batch = next(data_iter)
        learning_rate = 0.01
        optimizer = optim.Adam([{'params': [self.image_model.quaternion], 'lr': 1e-2 * learning_rate},
                                {'params': [self.image_model.coefficient], 'lr': learning_rate},
                                {'params': [self.image_model.bias], 'lr': 1e-2 * learning_rate}])
        criterion = nn.L1Loss()

        refer_img, test_img = batch['refer_img'][0].to(self.device), batch['test_img'][0].to(self.device)
        iters = tqdm(range(num_iters))

        for i in iters:
            self.optimizer.zero_grad()
            transformed_img, rotation_matrix = self.image_model(test_img)

            loss = criterion(transformed_img, refer_img)
            if self.lam[0] > 0:
                loss += self.lam[0] * torch.norm(rotation_matrix - torch.eye(3).to(self.device))
            for name, para in self.image_model.named_parameters():
                if self.lam[2] > 0 and 'bias' in name:
                    loss += self.lam[2] * para.norm(p=2)

            train_loss = loss.item()

            loss.backward()
            optimizer.step()

            if i == 0 or (i + 1) % 50 == 0:
                img = torch.cat([transformed_img[:, :, 40], refer_img[:, :, 40]], dim=1).cpu().detach().numpy()
                plt.imsave(self.root + f"/test/{i + 1:04d}.jpg", img, cmap='gray', vmin=0, vmax=1)

            iters.set_postfix({'loss': train_loss, 'lr': '%.4f' % self.optimizer.param_groups[0]['lr']})

        self.image_model.eval()
        transformed_img, _ = self.image_model(test_img)

        video_writer = imageio.get_writer(os.path.join(self.root, f'test/video.mp4'), fps=20)
        for z in range(transformed_img.shape[2]):
            img = torch.cat([test_img[:, :, z], transformed_img[:, :, z], refer_img[:, :, z]],
                            dim=1).cpu().detach().numpy()
            img = to8b(img)
            video_writer.append_data(img)

        # 关闭视频写入器
        video_writer.close()
        print('Done')
