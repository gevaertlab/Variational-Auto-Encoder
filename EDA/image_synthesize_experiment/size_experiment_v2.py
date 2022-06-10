import os
import os.path as osp
import sys
import torch
import numpy as np

sys.path.insert(1, '/labs/gevaertlab/users/yyhhli/code/vae/')


def lidc_experiment(log_name="VAE3D32AUG", version=70,
                    vis_dir="/labs/gevaertlab/users/yyhhli/code/vae/EDA/image_synthesize_experiment/results/"):
    # import lidc dataset
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    lidc_train = PATCH_DATASETS['LIDCPatchAugDataset'](
        root_dir=None, transform=sitk2tensor, split='train')
    lidc_val = PATCH_DATASETS['LIDCPatchAugDataset'](
        root_dir=None, transform=sitk2tensor, split='val')
    # get dataloaders
    from torch.utils.data.dataloader import DataLoader
    lidc_train_dataloader = DataLoader(
        dataset=lidc_train, batch_size=36, shuffle=False, drop_last=False, num_workers=4, pin_memory=False)
    lidc_val_dataloader = DataLoader(
        dataset=lidc_val, batch_size=36, shuffle=False, drop_last=False, num_workers=4, pin_memory=False)
    # import exporter
    from evaluations.export import Exporter
    exporter = Exporter(log_name=log_name, version=version,
                        dataloaders={"train": lidc_train_dataloader,
                                     "val": lidc_val_dataloader},
                        task_names=["volume"])
    # get data
    embeddings, data_names, label_dict = exporter.get_data()
    embeddings_train = np.array(embeddings["train"])
    volume = np.array(label_dict["volume"]['train'])  # numpy array

    # calculate the diff vector
    # select smallest and largest 5% nodules
    smallest_5_idx = volume.argsort()[:int(len(volume)*0.05)]
    largest_5_idx = volume.argsort()[-int(len(volume)*0.05):]
    smallest_5_embeddings = embeddings_train[smallest_5_idx]
    largest_5_embeddings = embeddings_train[largest_5_idx]
    smallest_5_embeddings_mean = smallest_5_embeddings.mean(axis=0)
    largest_5_embeddings_mean = largest_5_embeddings.mean(axis=0)
    diff_vector = largest_5_embeddings_mean - smallest_5_embeddings_mean

    # save the diff_vector
    # diff_vector_path = osp.join(vis_dir, "diff_vector.npy")
    # np.save(diff_vector_path, diff_vector)

    # select another batch of nodules (36 nodules) from train dataset
    median_idx = volume.argsort()[int(
        len(volume)*0.5)-18: int(len(volume)*0.5)+18]
    median_embeddings = embeddings_train[median_idx]

    # generate images
    from evaluations.evaluator import ReconEvaluator
    evaluator = ReconEvaluator(
        vis_dir=vis_dir, log_name=log_name, version=version)

    # experiment 1: enlarge and shrink the nodules
    import torch
    half_vector = torch.from_numpy(diff_vector[:2048]).type(
        torch.FloatTensor).to(evaluator.module.device) / np.sqrt(2)
    # enlarge
    enlarged_embeddings = torch.from_numpy(
        median_embeddings[:, :2048]) + half_vector
    # shrink
    shrinked_embeddings = torch.from_numpy(
        median_embeddings[:, :2048]) - half_vector

    median_images = evaluator.generate(torch.from_numpy(
        median_embeddings[:, :2048]).type(torch.float))
    enlarged_images = evaluator.generate(enlarged_embeddings.type(torch.float))
    shrinked_images = evaluator.generate(shrinked_embeddings.type(torch.float))

    # visualize all the images
    from utils.visualization import vis3d_tensor
    vis3d_tensor(median_images, save_path=osp.join(
        vis_dir, "test_nodules_median.jpeg"))
    vis3d_tensor(enlarged_images, save_path=osp.join(
        vis_dir, "test_nodules_enlarged.jpeg"))
    vis3d_tensor(shrinked_images, save_path=osp.join(
        vis_dir, "test_nodules_shrinked.jpeg"))

    # experiment 2: small to large nodules and large to small nodules
    small_embeddings = smallest_5_embeddings[:36,:2048]
    large_embeddings = largest_5_embeddings[:36,:2048]

    # convert large to small and small to large using vector
    s2l_embeddings = torch.from_numpy(small_embeddings) + half_vector * np.sqrt(2)
    l2s_embeddings = torch.from_numpy(large_embeddings) - half_vector * np.sqrt(2)

    # generate images
    small_images = evaluator.generate(torch.from_numpy(small_embeddings).type(torch.float))
    large_images = evaluator.generate(torch.from_numpy(large_embeddings).type(torch.float))
    s2l_images = evaluator.generate(s2l_embeddings.type(torch.float))
    l2s_images = evaluator.generate(l2s_embeddings.type(torch.float))

    # plot all images
    vis3d_tensor(small_images, save_path=osp.join(vis_dir, "test_nodules_small.jpeg"))
    vis3d_tensor(large_images, save_path=osp.join(vis_dir, "test_nodules_large.jpeg"))
    vis3d_tensor(s2l_images, save_path=osp.join(vis_dir, "test_nodules_s2l.jpeg"))
    vis3d_tensor(l2s_images, save_path=osp.join(vis_dir, "test_nodules_l2s.jpeg"))

    pass


def stf_experiment(log_name="VAE3D32AUG", version=70, task_name="StfVolume",
                   vis_dir="/labs/gevaertlab/users/yyhhli/code/vae/EDA/image_synthesize_experiment/results/"):
    # import stf dataset
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    train_ds = PATCH_DATASETS['StanfordRadiogenomicsPatchDataset'](
        root_dir=None, transform=sitk2tensor, split='train')
    val_ds = PATCH_DATASETS['StanfordRadiogenomicsPatchDataset'](
        root_dir=None, transform=sitk2tensor, split='val')
    # get dataloaders
    from torch.utils.data.dataloader import DataLoader
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=36,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False)
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=36,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False)
    # import exporter
    from evaluations.export import Exporter
    exporter = Exporter(log_name=log_name, version=version,
                        dataloaders={"train": train_dl,
                                     "val": val_dl},
                        task_names=[task_name])
    # get data
    embeddings, data_names, label_dict = exporter.get_data()
    embeddings_all = np.concatenate(
        [embeddings["train"], embeddings["val"]], axis=0)
    volume = np.array(label_dict[task_name]['train'] +
                      label_dict[task_name]['val'])  # numpy array

    # calculate the diff vector
    # select smallest and largest 10% nodules
    smallest_idx = volume.argsort()[:int(len(volume)*0.1)]
    largest_idx = volume.argsort()[-int(len(volume)*0.1):]
    smallest_embeddings = embeddings_all[smallest_idx]
    largest_embeddings = embeddings_all[largest_idx]
    smallest_embeddings_mean = smallest_embeddings.mean(axis=0)
    largest_embeddings_mean = largest_embeddings.mean(axis=0)
    diff_vector = largest_embeddings_mean - smallest_embeddings_mean

    # optional: load diff_vector
    diff_vector = np.load(osp.join(vis_dir, "diff_vector.npy"))

    # select another batch of nodules (36 nodules) from  dataset
    median_idx = volume.argsort()[int(
        len(volume)*0.5)-18: int(len(volume)*0.5)+18]
    median_embeddings = embeddings_all[median_idx]

    # generate images
    from evaluations.evaluator import ReconEvaluator
    evaluator = ReconEvaluator(
        vis_dir=vis_dir, log_name=log_name, version=version)

    # experiment 1: enlarge and shrink the nodules
    vector = torch.from_numpy(diff_vector[:2048]).type(
        torch.FloatTensor).to(evaluator.module.device) / np.sqrt(2)

    # vector = torch.from_numpy(diff_vector[:2048]).type(
    #     torch.FloatTensor).to(evaluator.module.device)
    # enlarge
    enlarged_embeddings = torch.from_numpy(
        median_embeddings[:, :2048]) + vector
    # shrink
    shrinked_embeddings = torch.from_numpy(
        median_embeddings[:, :2048]) - vector

    median_images = evaluator.generate(torch.from_numpy(
        median_embeddings[:, :2048]).type(torch.float))
    enlarged_images = evaluator.generate(enlarged_embeddings.type(torch.float))
    shrinked_images = evaluator.generate(shrinked_embeddings.type(torch.float))

    # visualize all the images
    from utils.visualization import vis3d_tensor
    vis3d_tensor(median_images, save_path=osp.join(
        vis_dir, "stf_nodules_median.jpeg"))
    vis3d_tensor(enlarged_images, save_path=osp.join(
        vis_dir, "stf_nodules_enlarged.jpeg"))
    vis3d_tensor(shrinked_images, save_path=osp.join(
        vis_dir, "stf_nodules_shrinked.jpeg"))
    # also visualize the largest and smallest nodules
    vis3d_tensor(evaluator.generate(torch.from_numpy(
        embeddings_all[largest_idx, :2048]).type(torch.float)), save_path=osp.join(
        vis_dir, "stf_largest_5_nodules.jpeg"))
    vis3d_tensor(evaluator.generate(torch.from_numpy(
        embeddings_all[smallest_idx, :2048]).type(torch.float)), save_path=osp.join(
        vis_dir, "stf_smallest_5_nodules.jpeg"))
    pass




if __name__ == "__main__":
    lidc_experiment()
