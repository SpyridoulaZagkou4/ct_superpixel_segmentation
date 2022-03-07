import os, glob
import torch
import numpy as np
import scipy.io
from skimage.color import rgb2lab
import matplotlib.pyplot as plt


def convert_label(label):
    onehot = np.zeros((1, 3, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0

    for t in np.unique(label).tolist():
        if ct >= 3:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


class ARTERY_Dataset:

    def __init__(self, preproc_dir, split='train', color_transforms=None, geo_transforms=None):
        self.preproc_dir = preproc_dir
        # self.img_shape = (512, 512)
        self.mode = split

        # self.img_files = sorted(glob.glob(os.path.join(preproc_dir, '*_img.npy')))
        # self.lab_files = sorted(glob.glob(os.path.join(preproc_dir, '*_lab.npy')))
        self.img_files = os.path.join(preproc_dir, split)
        self.lab_files = os.path.join(preproc_dir, split)

        self.help = sorted(glob.glob(os.path.join(preproc_dir, split, '*_img.npy')))

        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms

        assert len(self.img_files) == len(self.lab_files)

    def __getitem__(self, index):
        # pid = os.path.basename(self.img_files[index]).split('.')[0][:-4]
        # img = np.load(os.path.join(self.preproc_dir, pid + '_img.npy'))
        # lab = np.load(os.path.join(self.preproc_dir, pid + '_lab.npy'))

        pid = os.path.basename(self.help[index]).split('.')[0][:-4]
        img = np.load(os.path.join(self.img_files, pid + '_img.npy'))
        lab = np.load(os.path.join(self.lab_files, pid + '_lab.npy'))

        # if self.mode == 'train' and random.random() < self.augmentation_prob:
        #     img, lab = self.augment(img, lab)
        '''
        lab0 = np.argwhere(lab == 0)
        lab1 = np.argwhere(lab == 1)
        lab_exp = np.concatenate([np.zeros_like(lab[None, :]), np.zeros_like(lab[None, :])])
        lab_exp[0, lab0[:, 0], lab0[:, 1]] = 1
        lab_exp[1, lab1[:, 0], lab1[:, 1]] = 1

        '''
        lab = lab.astype(np.int64)
        img = img.astype(np.float32)

        lab = convert_label(lab)
        img = torch.from_numpy(img)
        lab = torch.from_numpy(lab)
        # img = img.permute(0, 1, 2)

        lab = lab.reshape(3, -1).float()

        return img, lab

    def __len__(self):
        return len(self.img_files)
