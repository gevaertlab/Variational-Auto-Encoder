# test stanford CT dataset
import os
import sys


def main():
    sys.path.append("/home/yyhhli/yyhhli/code/vae")
    from datasets import CT_DATASETS
    stf_ds = CT_DATASETS["StanfordRadiogenomicsDataset"](reset_info=True)
    stf_ds.register()


if __name__ == "__main__":
    main()
