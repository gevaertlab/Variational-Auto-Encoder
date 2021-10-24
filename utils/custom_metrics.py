from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import math
import torch


def ensure_numpy(img):
    return img.to("cpu").numpy() if torch.is_tensor(img) else img


def SSIM(orig, recon):
    orig = ensure_numpy(orig)
    recon = ensure_numpy(recon)
    return ssim(orig, recon, data_range=recon.max() - recon.min())


def MSE(orig, recon):
    orig = ensure_numpy(orig)
    recon = ensure_numpy(recon)
    return np.mean((orig - recon) ** 2)


def PSNR(orig, recon):  # peak signal to noise ratio
    orig = ensure_numpy(orig)
    recon = ensure_numpy(recon)
    mse = MSE(orig, recon)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = max(orig.max(), recon.max())
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_value
