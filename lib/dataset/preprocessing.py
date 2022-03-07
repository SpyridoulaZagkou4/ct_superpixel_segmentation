import os
import os.path as osp
from glob import glob
import nibabel as nib
import numpy as np
import random
import shutil
import skimage.transform as skTrans
from nibabel.processing import resample_to_output
import matplotlib.pyplot as plt

data_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/'

task_name = 'ArteriesCT'
preproc_TrDir = osp.join(data_dir, task_name, 'train')
preproc_TsDir = osp.join(data_dir, task_name, 'val')
# preproc_TsDir = osp.join(data_dir, task_name, 'test')
if osp.exists(preproc_TrDir): shutil.rmtree(preproc_TrDir)
if osp.exists(preproc_TsDir): shutil.rmtree(preproc_TsDir)
os.makedirs(preproc_TrDir, exist_ok=True)
os.makedirs(preproc_TsDir, exist_ok=True)

img_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/image'
lab_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/label'


def load_nifti(img_dir, lab_dir, pid):
    img_fn = osp.join(img_dir, pid + '.nii.gz')
    img = nib.load(img_fn)

    lab_fn = osp.join(lab_dir, pid + '.nii.gz')
    lab = nib.load(lab_fn)

    return img, lab


def find_mean_voxel_space():
    spacings = []
    shapes = []
    for i in range(len(fns)):
        lab = nib.load(fns[i])
        spac = [abs(lab.affine[0, 0]), abs(lab.affine[1, 1]), abs(lab.affine[2, 2])]
        spacings.append(spac)
        shapes.append(lab.shape)

    mean_spacing = np.mean(spacings, axis=0)

    target_spacing = mean_spacing
    mean_shape = [int(i) for i in np.mean(shapes, axis=0)]

    return target_spacing, mean_shape


def resample_crop(img, lab, target_spacing, box_size):
    # resample image and label to mean dataset spacing
    img_resampled = resample_to_output(img, voxel_sizes=target_spacing, order=1)
    lab_resampled = resample_to_output(lab, voxel_sizes=target_spacing, order=0, mode='nearest')

    return img_resampled, lab_resampled


def normalize(img_np, normalization='max_min'):
    if normalization == 'max_min':
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
    elif normalization == 'mean':
        mean_val, std_val = 0.0, 1.0
        img_np = (img_np - mean_val) / std_val

    return img_np


def resize(img, target_shape, type):
    # img_np = skTrans.resize(img_np, target_shape, preserve_range=True, mode= mode, order=order)
    if type == 'image':
        img_np = skTrans.resize(img, target_shape, order=1, preserve_range=True)
    if type == 'label':
        img_np = skTrans.resize(img, target_shape, order=0, mode='constant')

    return img_np


def save_slices(image, label, slice, pid, target_size, dir):
    (dimx, dimy, dimz) = image.shape
    cnt = 0
    if slice == "x":
        cnt += dimx
        for i in range(dimx):
            # np.save(osp.join(preproc_TrDir, pid + '_' + str(i) + '_img' + '.npy'),
            #         resize(image[i, :, :], target_size), allow_pickle=True)
            # np.save(osp.join(preproc_TrDir, pid + '_' + str(i) + '_lab' + '.npy'),
            #         resize(label[i, :, :], target_size), allow_pickle=True)
            np.save(osp.join(dir, pid + '_' + str(i) + '_img' + '.npy'),
                    resize(image[i, :, :], target_size, type='image'), allow_pickle=True)
            np.save(osp.join(dir, pid + '_' + str(i) + '_lab' + '.npy'),
                    resize(label[i, :, :], target_size, type='label'), allow_pickle=True)
    if slice == "y":
        cnt += dimy
        for i in range(dimy):
            # np.save(osp.join(preproc_TrDir, pid + '_' + str(i) + '_img' + '.npy'),
            #         resize(image[:, i, :], target_size), allow_pickle=True)
            # np.save(osp.join(preproc_TrDir, pid + '_' + str(i) + '_lab' + '.npy'),
            #         resize(label[:, i, :], target_size), allow_pickle=True)
            np.save(osp.join(dir, pid + '_' + str(i) + '_img' + '.npy'),
                    resize(image[:, i, :], target_size, type='image'), allow_pickle=True)
            np.save(osp.join(dir, pid + '_' + str(i) + '_lab' + '.npy'),
                    resize(label[:, i, :], target_size, type='label'), allow_pickle=True)
    if slice == "z":
        cnt += dimz
        for i in range(dimz):
            np.save(osp.join(preproc_TrDir, pid + '_' + str(i) + '_img' + '.npy'),
                    image[:, :, i], allow_pickle=True)
            np.save(osp.join(preproc_TrDir, pid + '_' + str(i) + '_lab' + '.npy'),
                    label[:, :, i], allow_pickle=True)
            #
            # plt.imsave(osp.join(dir, pid + '_' + str(i) + '_img' + '.jpg'), image[:, :, i])
            # plt.imsave(osp.join(dir, pid + '_' + str(i) + '_lab' + '.jpg'), label[:, :, i])


if __name__ == '__main__':
    fns = sorted(glob(osp.join(img_dir, '*')))
    # set data test proportion
    test_fraction = 0  # 0
    test_idx = random.sample(range(len(fns)), int(len(fns) * test_fraction))

    # find mean voxel space
    target_spacing, mean_shape = find_mean_voxel_space()

    # start preprocessing steps
    i = 0
    for i in range(len(fns)):
        pid = osp.basename(fns[i]).split('.')[0]
        print('\nPatient:', pid)

        # Load image
        img, lab = load_nifti(img_dir, lab_dir, pid)

        # Resample
        img_resampled, lab_resampled = resample_crop(img, lab, target_spacing, mean_shape)

        # Convert to npy
        img_np = np.squeeze(img_resampled.get_fdata(dtype=np.float32))
        lab_np = np.squeeze(lab_resampled.get_fdata(dtype=np.float32))

        # Normalize
        img_np = normalize(img_np)

        if i <= len(fns) * 0.7:
            # Save images
            save_slices(img_np, lab_np, 'z', pid, [128, 128], preproc_TrDir)
        else:
            save_slices(img_np, lab_np, 'z', pid, [128, 128], preproc_TsDir)
