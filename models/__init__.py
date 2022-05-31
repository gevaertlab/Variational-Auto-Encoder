from .vae_3d import VAE3D
from .vanilla_ae import VanillaAE
from .vae_3d_perceptual import VAE3DPerceptual

VAE_MODELS = {'VAE3D': VAE3D,
              'VAE3DPerceptual': VAE3DPerceptual,
              'VanillaAE': VanillaAE}