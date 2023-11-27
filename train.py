import numpy as np
import os
import nibabel as nib
import torch
from PIL import Image, ImageSequence
from dataparser.dataset import SDMDataset
from torch.utils.data import DataLoader
from model.statistical_deformable_model import SDM
from model.trainer import Trainer
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='SDM')
    # 添加命令行参数
    parser.add_argument('--load_path', help='Load SDM from ckpt path', default=None)
    # 解析命令行参数
    args = parser.parse_args()
    return args


def prepare_data_parser():
    # 使用前面定义的SDMDataset
    custom_dataset = SDMDataset()
    # 定义自己的数据加载器
    custom_dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=True, num_workers=4)
    return custom_dataset, custom_dataloader


def train(config):
    train_dataset, train_dataloader = prepare_data_parser()
    model = SDM(img_shape=(96, 80, 96), train_num=len(train_dataset), point_shape_factor=2)
    if config.load_path is None:
        print('No ckpt found, start training.')
        trainer = Trainer(model, train_dataset)
        trainer.train()
    else:
        print('Ckpt found.')
        checkpoint = torch.load(os.path.join(config.load_path, 'ckpt', 'SDM_weights.pth'))
        model.load_state_dict(checkpoint())
        print('Loaded!')
        trainer = Trainer(model, train_dataset, root=config.load_path)

    # trainer.eval()
    trainer.apply_pca(n_components = 18)
    trainer.analyse_test_image()


if __name__ == '__main__':
    config = arg_parser()
    train(config)
