from skimage.metrics import structural_similarity as ssim
import numpy as np
import math


def SSIM(orig, recon):
    return ssim(orig, recon, data_range=recon.max() - recon.min())


def MSE(orig, recon):
    return np.mean((orig - recon) ** 2)


def PSNR(orig, recon):  # peak signal to noise ratio
    mse = MSE(orig, recon)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = max(orig, recon.max())
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
