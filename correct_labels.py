import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

original_lab_dir = "/home/spyridoula/Downloads/CT_IMAGES/ct_original/label/"
superpix_lab_dir = "/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/Felzenswalb/superpixel_masks/"
dir_out = "/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/Felzenswalb/corrected/"

for f in sorted(os.listdir(original_lab_dir)):
    original_labels = nib.load(os.path.join(original_lab_dir, f))
    superpixels_labels = nib.load(os.path.join(superpix_lab_dir, f))

    original_labels_np = np.squeeze(original_labels.get_fdata())
    superpixels_labels_np = np.squeeze(superpixels_labels.get_fdata())


    lab1 = np.argwhere(original_labels_np == 1.0)
    lab2 = np.argwhere(original_labels_np == 2.0)

    for i in range(len(lab1)):
        superpixels_labels_np[lab1[i, 0], lab1[i, 1], lab1[i,2]] = 100

    for i in range(len(lab2)):
        superpixels_labels_np[lab2[i, 0], lab2[i, 1], lab2[i,2]] = 200

# plt.imshow(superpixels_labels_np[:,:, 100])
# plt.show()


# lab = np.load(os.path.join(lab_files, pid + '_lab.npy'))
# final = np.save(superpixels_labels_np)
    nib.save(nib.Nifti1Image(superpixels_labels_np, superpixels_labels.affine), os.path.join(dir_out, f))
