# test how patch_extract.py handles the dataset


def main():
    import sys
    import os
    import os.path as osp
    sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")
    from patch_extraction.patch_extract import PatchExtract
    from datasets import CT_DATASETS
    ds = CT_DATASETS["StanfordRadiogenomicsDataset"]()
    patch_extract = PatchExtract(patch_size=tuple([int(32)] * 3),
                                 dataset=ds,
                                 augmentation_params=None,
                                 debug=True)
    vlist = [24, 76, 42, 38,  2, 11, 85, 50, 77,  5, 51,  1, 56, 83, 45,  0, 44,
       48, 36, 43, 59, 25, 35, 19, 29, 94, 53, 49, 84,  9, 72, 71, 37, 13,
       20, 99]
    for i in vlist[:5]:
        patch_extract.load_extract(i, save_dir="/labs/gevaertlab/users/yyhhli/temp/patch_extract",
                                overwrite=True)
    patch_extract.vis_ds(dataset_dir="/labs/gevaertlab/users/yyhhli/temp/patch_extract",
                         vis_dir="/labs/gevaertlab/users/yyhhli/temp/patch_extract")
    pass

if __name__ == "__main__":
    main()
