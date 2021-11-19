# TEST FUNCTIONS

import sys
import os
import os.path as osp

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
    # task_names = ['volume', 'malignancy', 'texture', 'spiculation', 'subtlety']
    task_names = ['volume']
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


def test_synth():
    from evaluations.evaluator import SynthesisGaussian
    synth = SynthesisGaussian(log_name='VAE3D32AUG',
                              version=18,
                              vis_dir=osp.join(os.getcwd(), "evaluations/results/"))
    synth.synth_and_vis()
    pass


def test_reparametrizations():
    from evaluations.evaluator import SynthsisReParam
    reparam = SynthsisReParam(log_name='VAE3D32AUG',
                              version=18,
                              vis_dir=osp.join(os.getcwd(), "evaluations/results/"))
    reparam()
    pass


def test_value_range():
    from evaluations.evaluator import SynthesisRange
    synth_range = SynthesisRange(log_name='VAE3D32AUG',
                                 version=18,
                                 vis_dir=osp.join(os.getcwd(), "evaluations/results/"))
    synth_range()
    pass


def test_value_range_v2():
    volume_feature = [
        1533, 2029,  401,  129,  283, 1128, 2038,    7,  469,  896, 1704,
        65, 1680, 1167,  496, 1254,  909,  138, 1095,   55, 1025, 1692,
        153,  733, 1898,  434,  107,  598, 1130, 1269,  556,  832,  305,
        1549, 2001,  806,  754, 1530,  957, 1009, 1163, 1838,  328, 1927,
        856, 1333, 1211, 1894, 1448, 1252,  369,  333, 1764, 1625,  376,
        121, 1417, 1649,  741, 1933,  262,  926, 1196, 1429,  734,  551,
        175,  884, 1846,  145, 1560,  188, 1118,  448,   59, 1436, 1872,
        721, 2008,  841,  871,  602, 1296,  622, 1000, 1028,  559, 1805,
        534,   88,  416, 1777,   66,  115,  302,   80,  181, 1987, 1329,
        365, 1404, 1520, 1595,  791,  899,  581,  987,  299, 1899,  274,
        276,  828,  972, 1811,  595,  264, 1978, 1127,  857, 1631, 1691,
        533, 1626,  410,  489,  756,  169,  880, 1936, 1178, 1123,   25,
        1022, 1342, 1561, 1086,  335, 1011, 1013,  813,  224, 1693,  191,
        611,  523, 1885,  185,  915,    9,   76, 1780,  104,  151, 1019,
        241,  815, 1464, 1422, 1897,  874, 1365, 1870,  946, 1402, 1239,
        113, 1267, 1054,  243,  554, 2026, 1584,  508,  311,  983,  691,
        481, 1585, 1225, 1940,  701, 1970,  495,  859, 1007,  318, 1613,
        1736, 1776, 1855,  700,  503,  136,  105, 1389, 1089,  799,  260,
        1245, 1738, 1270,  522, 1975,  494,  342,  337,  540,  947, 1818,
        560, 1014,  200,  738, 1190, 1695, 1718,  440,  589, 1778,  371,
        164,  388,  830, 1583,  400, 1063,  888, 1229,  372,  419,  111,
        882, 1223, 1360, 1878, 1837,  366, 1610,  782,  295,  903,  811,
        538, 1739, 1661,  867, 2035,   10, 2032, 1650, 1421, 1591,  636,
        230,  986,  511, 1112,   72, 1828, 1151, 1606,  747, 1131, 1526,
        1347, 1427,    5, 1292,   91, 1466, 1506, 1253,  392,  477,   98,
        1831, 1497, 1912,  502,  439, 1459,  118,  924, 1449,  491, 1420,
        1679, 1099,  285, 1767, 1803,  380,  846,  796, 1336, 1455,  122,
        1107,  711, 1556, 1700, 1132, 1926, 1156, 1235, 1873, 1236,   92,
        1961,  643, 1104, 1749,  148, 1034,  529,  773,  712, 1913, 1310,
        977, 1640, 1385, 1588, 1173,  940, 1231,  332, 1492, 2012, 1180,
        790,    6, 1518, 1265, 1387, 1101, 1377, 1521,  336, 1332,  990,
        1616,  450, 1411, 1056, 1558,  382,   17,  730, 1401, 1548, 1972,
        1147,  764,  339,  984, 2045,  710,   68, 1623, 1479, 1980,  642,
        205,  607,  878,  965, 1396,  613, 1573, 1343, 1072, 1787,  134,
        2013,  869,  356, 1451, 1240,  562,  317,  904, 1839,  837, 1931,
        1314,  214, 1596,  127,  911,  657, 1949, 1188,  350, 1834,  180,
        1804, 1998,  695,  669,  429,  242, 1937, 1713,  749,  287,  802,
        132, 1046, 1207, 1864,  820,  480, 1142, 1291,  864, 1049, 1283,
        678,  437,   62, 1380, 1808,  319, 1771,   79, 2010, 1666, 1301,
        1328, 1773,  762,  485,  249, 1725,  514, 1617, 1060, 1901,  231,
        875, 1340,  506, 1029,  935,  680, 1088, 1824, 1501, 1712,  500,
        1469
    ]
    from evaluations.evaluator import SynthesisRange
    synth_range = SynthesisRange(log_name='VAE3D32AUG',
                                 version=18,
                                 vis_dir=osp.join(os.getcwd(), "evaluations/results/"))
    synth_range(feature_idx=volume_feature)
    pass


if __name__ == "__main__":
    test_value_range_v2()
