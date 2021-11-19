## 3D Variational-Auto-Encoder for Lung Nodule CTs

### OVERVIEW
This is a project for training and validating a patch-based lung nodule 3D variational auto-encoder. To train a model, you need to first preprocess a CT dataset to get nodule patches using `preprocessing.py`, and then train the model with `train.py`.

### FIRST STEPS
There are several things to modify if you  want to run a demo on your machine. First, in `configs/config_vars.py`, change the `BASE_DIR` and `DS_ROOT_DIR` to your code root directory and your dataset root directory respectively. Secondly, in `dataset` folder add your CT, patch and label dataset implementations with the examples and base classes implemented.


### TRAINING A DEMO MODEL
[LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) dataset and [LNDb](https://lndb.grand-challenge.org/Data/) dataset are default datasets that are already implemented. If you want to run a demo model using LIDC or LNDb dataset, place downloaded LIDC or LNDb dataset under your `DS_ROOT_DIR` and run `preprocessing.py` with appropriate parameters. After getting the patch dataset, you can run `train.py` to train a VAE model, to change the settings of training, create a `yml` file in `configs` folder, refer to the examples there.


### EVALUATING THE MODEL
The model can be validated with evaluators in `evaluations/evaluator.py`, with detailed descriptions. Downstream models can be trained and tested with `applications/application.py`, including prediction models and association tests.

### EXAMPLE CODE

In `EDA` folder, multiple example code were provided, which you can refer to for different tasks. `test_datasets.py` shows how CT and patch datasets are used, and `test_eval.py` shows the functions and usage of different evaluators.


### CREDICT & CONTACT

This is an unpublished work by Yiheng Li at [Gevaert Lab](https://med.stanford.edu/gevaertlab.html) of [Stanford DBDS](https://med.stanford.edu/dbds.html). If you have any question regarding the code, please contact Yiheng (Terry) Li via email yyhhli@stanford.edu or [Dr. Olivier Gevaert](https://profiles.stanford.edu/olivier-gevaert).