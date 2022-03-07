import math
import numpy as np
import torch

from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter
import nibabel as nib

@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None, enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """
    if weight is not None:
        from model import SSNModel
        model = SSNModel(fdim, nspix, n_iter).to("cuda")
        model.load_state_dict(torch.load(weight))
        model.eval()
    else:
        model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    height, width = image.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

    # [1, 2, 321, 481]
    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()

    # image = rgb2lab(image)
    # [1, 3, 321, 481]
    image = torch.from_numpy(image)[None].to("cuda").float()
    image=image.unsqueeze(0)
    # image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    # print(image.size())
    # print(coords.size())

    inputs = torch.cat([color_scale*image, pos_scale*coords], 1)

    _, H, _ = model(inputs)

    labels = H.reshape(height, width).to("cpu").detach().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=20, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.78, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()

    # image = plt.imread(args.image)
    #
    # s = time.time()
    # label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
    # print(f"time {time.time() - s}sec")
    # plt.imsave("results.png", mark_boundaries(image, label))

    # image = plt.imread(args.image)
    img = nib.load(args.image)
    image = img.get_fdata()

    s = time.time()
    label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
    print(f"time {time.time() - s}sec")

    new_img = mark_boundaries(image,label)

    # plt.imsave("results.png", new_image)

    # plt.imshow(new_img)
    plt.imshow((new_img * 255).astype(np.uint8))
    plt.show()

    # telikh_image= nib.Nifti1Image(label)
    # save_path="/home/spyridoula/Downloads/Self-supervised-Fewshot-Medical-Image-Segmentation/Unsupervised_Superpixels/output/1.nii.gz"
    # nib.save(telikh_image, save_path)
