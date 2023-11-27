import os
import nibabel as nib
import torch
from torch.utils.data import Dataset


class SDMDataset(Dataset):
    def __init__(self, filepath='./datasets/affregcommon2mm_roi_ct_train'):
        print(f'Init Dataset from {filepath}')
        # 读取本代码同个文件夹下所有的nii格式的文件
        self.filenames = sorted(os.listdir(filepath))
        self.labels = []
        self.img = []  # [refer 1, test 1, train n-2]
        self.label = []  # [refer 1, test 1, train n-2]

        # 读入带label与图像
        for f in self.filenames:
            imgs = nib.load(os.path.join(filepath, f))
            self.shape = imgs.shape
            h, w, z = imgs.shape
            fdata = imgs.get_fdata()

            if len(self.labels) == 0 and 'label' in f:
                for i in range(h):
                    for j in range(w):
                        for k in range(z):
                            if fdata[i, j, k] not in self.labels:
                                self.labels.append(fdata[i, j, k])

                self.labels.sort()
                print(f'labels: {self.labels}')

            if 'label' in f:
                self.label.append(fdata)
            else:
                fdata[fdata == 0] = -1024
                fdata = (fdata + 1024) / 4095  # 像素值归一化
                self.img.append(fdata)
                # print(fdata.min(), fdata.max())

            # if len(self.label) == 4 and len(self.img) == 4:
            #     return

    def __len__(self):
        return len(self.img) - 2

    def __getitem__(self, idx):
        sample = {
            'idx': idx,
            'refer_img': self.img[0],
            'refer_label': self.label[0],
            'test_img': self.img[1],
            'test_label': self.label[1],
            'train_img': self.img[idx + 2],
            'train_label': self.label[idx + 2],
            'label_name': self.labels
        }

        return sample

    def video(self):
        # images = []
        # for zi in range(z):
        #     x = Image.fromarray(fdata[:, :, zi])
        #     images.append(x)
        # images[0].save("test_z.gif", save_all=True, loop=True, append_images=images[1:], duration=100)
        raise NotImplementedError
