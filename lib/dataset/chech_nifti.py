import numpy as np
import os.path as osp
import nibabel as nib

data_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/'

task_name = 'ArteriesCT'
preproc_TrDir = osp.join(data_dir, task_name, 'inference_image')


# preproc_TsDir = osp.join(data_dir, task_name,

def save_nifti(img_np, lab_np, img, lab):
    nib.save(nib.Nifti1Image(img_np, img.affine), osp.join(preproc_TrDir, 'my_check_img.nii.gz'))
    nib.save(nib.Nifti1Image(lab_np, lab.affine), osp.join(preproc_TrDir, 'my_check_lab.nii.gz'))


if __name__ == '__main__':
    img_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/image/0041.nii.gz'
    lab_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/label/0041.nii.gz'

    img_npy_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/ArteriesCT/train/0041_150_img.npy'
    lab_npy_dir = '/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/ArteriesCT/train/0041_150_lab.npy'

    img_npy = np.load(img_npy_dir)
    lab_npy = np.load(lab_npy_dir)

    img, lab = nib.load(img_dir), nib.load(lab_dir)

    save_nifti(img_npy, lab_npy, img, lab)