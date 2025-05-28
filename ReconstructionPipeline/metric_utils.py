from skimage.morphology import skeletonize, skeletonize_3d
from skimage.metrics import structural_similarity
import numpy as np
import torch

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    return 2*tprec*tsens/(tprec+tsens)


def get_ssim_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
        Format-[N H W D], OriImage [0,1]
    :param arr2:
        Format-[N H W D], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    N = arr1.shape[0]
    # Depth
    arr1_d = arr1#np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = arr2#np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = structural_similarity(arr1_d[i], arr2_d[i], data_range=2)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float32)

    # Height
    arr1_h = np.transpose(arr1, (0, 2, 3, 1))
    arr2_h = np.transpose(arr2, (0, 2, 3, 1))
    ssim_h = []
    for i in range(N):
        ssim = structural_similarity(arr1_h[i], arr2_h[i], data_range=2)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float32)

    # Width
    arr1_w = np.transpose(arr1, (0, 3, 1, 2))
    arr2_w = np.transpose(arr2, (0, 3, 1, 2))
    ssim_w = []
    for i in range(N):
        ssim = structural_similarity(arr1[i], arr2[i], data_range=2)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float32)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg

def get_psnr_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    """
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # #zero mse, return 100
    psnr[zero_mse] = 100

    if size_average:
        return psnr.mean()
    else:
        return psnr