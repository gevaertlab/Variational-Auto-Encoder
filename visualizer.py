""" provide functionalities for visualizing results using functions in visualization file """

from utils.custom_loggers import VAELogger

if __name__ == '__main__':
    vae_logger = VAELogger(
        save_dir="logs/",
        name="VAE3D32",
        version=64
    )
    vae_logger.draw_loss_curve()
    vae_logger.draw_kl_recon_loss()
    vae_logger.draw_multiple_loss_curves()
