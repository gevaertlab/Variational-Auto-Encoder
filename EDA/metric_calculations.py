from evaluations.evaluator import MetricEvaluator
# TODO: run this


if __name__ == '__main__':
    me = MetricEvaluator(metrics=['SSIM', 'MSE', 'PSNR'],
                         log_name='VAE3D32AUG',
                         version=10)
    me.calc_metrics()
    pass
