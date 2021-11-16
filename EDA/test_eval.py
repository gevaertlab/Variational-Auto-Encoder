# TEST FUNCTIONS

import sys
import os

from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, os.getcwd())


def test_metrics_calculation_for_lndb_dataset():
    from datasets import PATCH_DATASETS
    import numpy as np
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='val')
    print("length of lndb_patch dataset", len(lndb_patch))
    lndb_dl = DataLoader(dataset=lndb_patch,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    me = MetricEvaluator(metrics=["SSIM", "MSE", "PSNR"],
                         log_name='VAE3D32AUG',
                         version=18)
    metrics_dict = me.calc_metrics(dataloader=lndb_dl)
    for k, v in metrics_dict.items():
        print(f"{k}: mean value = {np.mean(v)}")
    pass


def test_recon_images_for_lndb_dataset():
    from datasets import PATCH_DATASETS
    import os.path as osp
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import ReconEvaluator
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='val')
    print("length of lndb_patch dataset", len(lndb_patch))
    lndb_dl = DataLoader(dataset=lndb_patch,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    re = ReconEvaluator(vis_dir=osp.join(os.getcwd(), "evaluations/results/"),
                        log_name='VAE3D32AUG',
                        version=18)
    re(dataloader=lndb_dl)
    pass


def test_lidc_model_on_lndb_dataset():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from datasets import LNDbDataset
    from applications.application import Application

    lndb = LNDbDataset()

    lndb_train_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                            transform=sitk2tensor,
                                                            split='train')
    lndb_train_patch_dataloader = DataLoader(dataset=lndb_train_patch,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=4,
                                             pin_memory=True)
    lndb_val_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                          transform=sitk2tensor,
                                                          split='val')
    lndb_val_patch_dataloader = DataLoader(dataset=lndb_val_patch,
                                           batch_size=1,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=4,
                                           pin_memory=True)
    app = Application(log_name='VAE3D32AUG',
                      version=18,
                      task_name='LNDbTaskVolume',
                      base_model_name='VAE3D',
                      dataloader={'train': lndb_train_patch_dataloader,
                                  'val': lndb_val_patch_dataloader})

    result_dict, pred_dict, pred_stats, hparam_dict = app.task_prediction(
        tune_hparams=False, models='all')

    return result_dict, pred_dict, pred_stats, hparam_dict


def test_association_analysis():
    from applications.application import Application
    result_dict = {}
    task_names = ['volume', 'malignancy', 'texture', 'spiculation', 'subtlety']
    for task_name in task_names:
        app = Application(log_name='VAE3D32AUG',
                          version=18,
                          task_name=task_name)
        sig_df = app.association_analysis()
        result_dict[task_name] = sig_df
    return result_dict


def test_feature_selection_lidc():
    from applications.application import Application
    result_dict = {}
    task_names = ['volume', 'malignancy', 'texture', 'spiculation', 'subtlety']
    for task_name in task_names:
        app = Application(log_name='VAE3D32AUG',
                          version=18,
                          task_name=task_name)
        result = app.task_prediction(tune_hparams=False)
        result_dict[task_name] = result
        app.draw_best_figure()
    return result_dict


def test_visualizations_lidc():
    from applications.application import Application
    result_dict = {}
    task_names = ['volume', 'malignancy', 'texture', 'spiculation', 'subtlety']
    for task_name in task_names:
        app = Application(log_name='VAE3D32AUG',
                          version=18,
                          task_name=task_name)
        result = app.task_prediction(tune_hparams=False)
        result_dict[task_name] = result
        # app.visualize()
    return result_dict


def test_association_analysis_on_lndb_dataset():
    from applications.application import Application
    result_dict = {}
    task_names = ['volume', 'malignancy', 'texture', 'spiculation', 'subtlety']
    for task_name in task_names:
        app = Application(log_name='VAE3D32AUG',
                          version=18,
                          task_name=task_name)
        sig_df = app.association_analysis()
        result_dict[task_name] = sig_df
    return result_dict


def test_recon_images():
    from datasets import PATCH_DATASETS
    import os.path as osp
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import ReconEvaluator
    lidc_patch = PATCH_DATASETS["LIDCPatchAugDataset"](root_dir=None,
                                                       transform=sitk2tensor,
                                                       split='val')
    print("length of lidc_patch dataset", len(lidc_patch))
    lndb_dl = DataLoader(dataset=lidc_patch,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    re = ReconEvaluator(vis_dir=osp.join(os.getcwd(), "evaluations/results/"),
                        log_name='VAE3D32AUG',
                        version=18)
    re(dataloader=lndb_dl, num_batches=1)
    pass


def test_exporter():
    from evaluations.export import Exporter
    exporter = Exporter(log_name='VAE3D32AUG',
                        version=18,
                        task_names=['volume'],
                        )
    embeddings, data_names, label_dict = exporter.get_data()
    embeddings_train = np.array(embeddings['train'])
    plt.hist(embeddings_train.flatten(), bins=200, density=True)
    plt.savefig("/labs/gevaertlab/users/yyhhli/temp/all_dens.jpeg", dpi=300)
    plt.close()
    plt.hist(embeddings_train[:, :2048].flatten(), bins=200, density=True)
    plt.savefig("/labs/gevaertlab/users/yyhhli/temp/mean_dens.jpeg", dpi=300)
    plt.close()
    plt.hist(embeddings_train[:, 2048:].flatten(), bins=200, density=True)
    plt.savefig("/labs/gevaertlab/users/yyhhli/temp/std_dens.jpeg", dpi=300)
    plt.close()
    pass


if __name__ == "__main__":
    test_exporter()
