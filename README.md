This project was based on the PyTorch implementation of Superpixel Sampling Networks (SSN) https://github.com/perrying/ssn-pytorch

paper: https://arxiv.org/abs/1807.10174

original code: https://github.com/NVlabs/ssn_superpixels


With some customizations I applied this framework to gray-scale CT images in order to apply automatic segmentation to aorta and arteries. 

# REQUIREMENTS
1. PyTorch >= 1.4
2. scikit-image
3. matplotlib

# USAGE
## inference
python inference.py --image ~/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/ArteriesCT/inferenc
e_image/my_check_img.nii.gz --weight ~/Downloads/ssn-pytorch/log/model1645611288.pth

## train 
python train.py --root ~/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/ArteriesCT/


# RESULTS



![image](https://user-images.githubusercontent.com/81852029/204155125-10f3fa29-a85f-4863-981a-cf093f3996e4.png)
![image](https://user-images.githubusercontent.com/81852029/204155112-6a83789d-eae2-4d37-9218-2ce5e9c0b123.png)




![image](https://user-images.githubusercontent.com/81852029/204155890-412343cf-0d9e-477f-a1e2-aa5054934ece.png)
![image](https://user-images.githubusercontent.com/81852029/204155776-c94e7c37-c9e5-4e49-97fa-bb2be55d2b96.png)

