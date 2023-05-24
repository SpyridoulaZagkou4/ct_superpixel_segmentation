# DESCRIPTION

Development of a deep superpixel-based graph cut network for weakly supervised semantic segmentation of 3D CT images. Training of Deep Neural Networks (U-Net encoder) to extract biomedical image features, applying a KMeans-type algorithm by averaging these features to initialize super-pixel centers and then recalculating them iteratively using association map distance among neighboring (super)pixels


### reference
This project was based on the PyTorch implementation of Superpixel Sampling Networks (SSN) https://github.com/perrying/ssn-pytorch

paper: https://arxiv.org/abs/1807.10174

original code: https://github.com/NVlabs/ssn_superpixels

With some customizations I applied this framework to gray-scale CT images in order to apply automatic segmentation for aorta and arteries. 

### steps
First run ```felzenswalb.py``` script in the label_production folder to extract enhanced segmentations and then run ```correct_labels.py``` to assure that ROIs are correct. Then you have to run ```preprocessing.py``` script in lib\dataset forder to save imager to .npy format. Finally you are ready to train and test your model. 
(see usage below).

# REQUIREMENTS
1. PyTorch >= 1.4
2. scikit-image
3. matplotlib
4. nibabel
5. numpy
6. scipy
7. skimage

I run the code to Windows machine system and I had installed WSL to embed linux system that is suggested from the original code.
* cuda required
* ninja required

# USAGE
## inference
```
python inference.py --image ~/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/ArteriesCT/inferenc
e_image/my_check_img.nii.gz --weight ~/Downloads/ssn-pytorch/log/model1645611288.pth
```

## train 
```
python train.py --root ~/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/ArteriesCT/
```


# RESULTS



![image](https://user-images.githubusercontent.com/81852029/204155125-10f3fa29-a85f-4863-981a-cf093f3996e4.png)
![image](https://user-images.githubusercontent.com/81852029/204155112-6a83789d-eae2-4d37-9218-2ce5e9c0b123.png)



![image](https://user-images.githubusercontent.com/81852029/204475836-136187c9-8e75-4f85-89d7-21bbd89e056e.png)
![image](https://user-images.githubusercontent.com/81852029/204475859-ce04784c-680d-4257-8ea5-88b196344ecd.png)

