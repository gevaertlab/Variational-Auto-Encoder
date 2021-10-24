import numpy as np
from evaluations.evaluator import MetricEvaluator
from train import main
from applications.application import Application

if __name__ == '__main__':
    exp_lst = ['exp', 'default', 'lparams',
               'xlparams', 'ldim',  'lbeta', 'sbeta']
    for i, exp_name in enumerate(exp_lst):
        # main(f"exp_new/vae32aug_{exp_name}")
        # me = MetricEvaluator(metrics=['SSIM', 'MSE', 'PSNR'],
        #                      log_name='VAE3D32AUG',
        #                      version=11 + i,
        #                      base_model_name='VAE3D')
        # metrics_dict = me.calc_metrics()
        # for k, v in metrics_dict.items():
        #     print(f"{k}: mean value = {np.mean(v)}")

        # downstream task prediction
        app = Application(log_name="VAE3D32AUG",
                          version=11 + i,
                          task_name="volume")
        app.task_prediction(tune_hparams=False, models="all")
        app.save_results()
        app.draw_dignosis_figure()
    pass
