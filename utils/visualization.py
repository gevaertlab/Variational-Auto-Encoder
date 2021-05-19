''' This file provides util functions to visualize various type of data '''
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torchvision.utils as vutils
import os
import math
import pandas as pd
from typing import Dict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
import math
plt.style.use('ggplot')


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def visImg(img_array, rng=(0, 1), temp_dir='/home/yyhhli/code/image data/temp_img.png'):
    plt.imsave(temp_dir, img_array.astype(np.float), vmin=rng[0], vmax=rng[1])
    # print("Image visualized at", temp_dir)
    plt.close()
    pass


def vis3D(img, axis=0, slice_num=None, temp_dir='/home/yyhhli/code/image data/temp_img.png'):
    if slice_num is None:
        slice_num = int(img.shape[axis]/2)
    indices = {0: None, 1: None, 2: None}
    indices[axis] = slice_num
    # print(img[tuple(slice(indices[i]) if indices[i] is None else indices[i] for i in range(3))].shape)
    visImg(img[tuple(slice(indices[i]) if indices[i] is None else indices[i]
                     for i in range(3))], temp_dir=temp_dir)
    pass


def vis3DTensor(img_tensor, axis=0, slice_num=None, save_dir='/home/yyhhli/code/image data/temp_img.png'):
    ''' 
    Visualize image tensor of a batch 
    @param: img_tensor: [B, C, L, W, H]
    @axis: select [L, W, H] to cut the slice, default to be 0 -- L
    @slice_num: slice number to cut, default to be the middle layer
    '''

    # using vutils
    if slice_num is None:
        slice_num = int(img_tensor.size()[axis + 2]/2)
    indices = {0: None, 1: None, 2: None, 3: None, 4: None}
    indices[axis + 2] = slice_num
    img_tensor_slice = img_tensor[tuple(
        slice(indices[i]) if indices[i] is None else indices[i] for i in range(5))]
    vutils.save_image(img_tensor_slice.data, save_dir, normalize=True, nrow=8)


def visSitk(img, axis=2, slice_num=None, temp_dir='/home/yyhhli/code/image data/temp_img.png'):
    axis2axis = {0: 1, 1: 2, 2: 0}
    np_img = sitk.GetArrayFromImage(img)
    new_axis = axis2axis[axis]
    vis3D(np_img, axis=new_axis, slice_num=slice_num, temp_dir=temp_dir)


def vis_loss_curve(log_path: str, data: Dict):

    # draw loss curve
    fig = plt.figure(figsize=(6, 4))
    plt.yscale('log')
    for key, value in data.items():
        plt.plot(value['epoch'], value['loss'], label=key)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    loss_lst = flatten_list([list(value['loss'])
                             for key, value in data.items()])
    plt.ylim(np.min(loss_lst),
             np.percentile(loss_lst, 98))
    plt.legend()
    fig.savefig(os.path.join(log_path, 'loss_csv.png'))
    plt.close()


def vis_loss_curve_diff_scale(log_path: str, data: Dict):
    """ draw loss curve with two diff scales """
    fig, ax1 = plt.subplots(figsize=(6, 4))
    plt.yscale('log')
    # extract keys, assert only two
    keys = data.keys()
    assert len(keys) == 2

    # plot the first one
    ax1.set_xlabel('epoch')
    ax1.set_ylabel(keys[0])
    ax1.plot(data[keys[0]]['epoch'], data[keys[0]]['loss'])
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # plot the second one
    ax2 = ax1.twinx()

    ax2.set_xlabel('epoch')
    ax2.set_ylabel(keys[1])
    ax2.plot(data[keys[1]]['epoch'], data[keys[1]]['loss'])
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.savefig(log_path)
    plt.close()


def residual_plot(y_true, y_pred, save_dir='/home/yyhhli/temp.jpeg'):
    plt.scatter(y_true - y_pred, y_true)
    plt.savefig(save_dir)
    plt.close()
    pass


def ytrue_ypred_scatter(pred_dict, save_dir='/home/yyhhli/temp.jpeg'):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    y_true = pred_dict['true']
    ax.set_aspect('equal')
    for model_name, pred in pred_dict.items():
        if model_name != "true":
            plt.scatter(y_true, pred, label=model_name)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, linestyle='dashed', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300)
    print(f"plotted at {save_dir}")
    plt.close()


def plot_cm(y_true,
            y_pred,
            save_dir,
            title=None,
            classes=list(range(1, 6)),
            cmap=plt.cm.Blues):
    """
    Draw confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.grid(False)
    plt.savefig(save_dir, dpi=400)
    plt.close()
    print(f"Visualized at {save_dir}")
    pass


def confusion_matrix_subplot(y_true,
                             y_pred,
                             ax,
                             title='',
                             classes=list(range(1, 6)),
                             cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap=cmap, ax=ax)
    ax.title.set_text(title)
    ax.grid(False)
    pass


def confusion_matrix_models(pred_dict, save_dir, classes=list(range(1, 6))):
    model_names = list(pred_dict.keys())
    model_names.remove('true')
    y_true = pred_dict['true']
    if len(pred_dict.keys()) == 2:
        # only one model
        plot_cm(y_true,
                pred_dict[model_names[0]],
                save_dir=save_dir,
                title=model_names[0],
                classes=list(range(1, 6)))
        pass
    else:
        # multiple models
        fig, axs = plt.subplots(len(pred_dict.keys()) - 1,
                                1, figsize=(5, 5 * len(pred_dict.keys()) - 1))
        for i in range(len(pred_dict.keys()) - 1):
            ax = axs[i]
            y_pred = pred_dict[model_names[i]]
            confusion_matrix_subplot(y_true,
                                     y_pred,
                                     ax=ax,
                                     title=model_names[i],
                                     classes=classes)
        plt.savefig(save_dir, dpi=400)
        plt.close()
        print(f"Visualized at {save_dir}")
        pass
