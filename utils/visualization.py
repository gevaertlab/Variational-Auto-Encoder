""" This file provides util functions to visualize various type of data  """

import math
import os
import os.path as osp
from typing import Dict, Union

# import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
# import torchvision.utils as vutils
from umap.umap_ import UMAP
# import umap.plot
from matplotlib.patches import Patch
# from matplotlib.ticker import MultipleLocator, ScalarFormatter
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from statannot import add_stat_annotation

from utils.python_logger import get_logger

from .timer import Timer


plt.style.use('ggplot')
plt.ioff()  # Turn off interactive mode
matplotlib.use('Agg')

LOGGER = get_logger()


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


def vis_img(img_array,
            rng=(0, 1),
            vis_path='/home/yyhhli/code/image data/temp_img.png'):

    plt.imsave(vis_path, img_array.astype(np.float), vmin=rng[0], vmax=rng[1])
    plt.close()
    pass


def vis_img_with_point(img: np.ndarray,
                       point_coord_dict: dict,
                       vis_path: str):
    """visualize image with points on the image

    Args:
        img ([np.ndarray]): [image to visualize]
        point_coord_dict ([dict]): e.g. {"name(label)": (x, y)}
        vis_path ([str]): [path to save image]
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    for key, value in point_coord_dict.items():
        ax.plot(value)
        ax.annotate(key, value)
    plt.savefig(vis_path, dpi=300)
    pass


def vis3d(img,
          axis=2,
          slice_num=None,
          vis_path='/home/yyhhli/code/image data/temp_img.png'):
    if slice_num is None:
        slice_num = int(img.shape[axis]/2)
    indices = {0: None, 1: None, 2: None}
    indices[axis] = slice_num
    vis_img(img[tuple(slice(indices[i]) if indices[i] is None else indices[i]
                      for i in range(3))], vis_path=vis_path)
    pass


def vis3d_tensor(img_tensor,
                 axis=0,
                 slice_num=None,
                 nrow=6,
                 save_path=None):
    '''
    Visualize image tensor of a batch 
    @param: img_tensor: [B, C, L, W, H]
    @axis: select [L, W, H] to select the slice, default to be 0 -- L
    @slice_num: slice number to select, default to be the middle layer
    '''

    # using vutils
    if slice_num is None:
        slice_num = int(img_tensor.shape[axis + 2]/2)
    indices = {0: None, 1: None, 2: None, 3: None, 4: None}
    indices[axis + 2] = slice_num
    img_tensor_slice = img_tensor[tuple(
        slice(indices[i]) if indices[i] is None else indices[i] for i in range(5))]
    img_tensor_slice = img_tensor_slice.data.numpy()
    # vutils.save_image(img_tensor_slice.data,
    #                   save_path,
    #                   normalize=True,
    #                   nrow=nrow)

    # using matplotlib
    # img_tensor_slice.shape = (B, C=1, W, H)
    ncols = int(math.ceil(img_tensor_slice.shape[0]/nrow))
    fig, axes = plt.subplots(nrows=nrow,
                             ncols=ncols,
                             figsize=(3*ncols, 3*nrow))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img_tensor_slice[i, 0, :, :], cmap='gray')
        ax.axis('off')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=400)
        LOGGER.info('save image to {}'.format(save_path))
    else:
        plt.show()
    plt.close()
    pass


def vis_sitk(img, axis=2, slice_num=None, vis_path='/home/yyhhli/code/image data/temp_img.png'):
    axis2axis = {0: 1, 1: 2, 2: 0}
    np_img = sitk.GetArrayFromImage(img)
    new_axis = axis2axis[axis]
    vis3d(np_img,
          axis=new_axis,
          slice_num=slice_num,
          vis_path=vis_path)


def vis_loss_curve(log_path: str, data: Dict, name='loss_curve.jpeg'):
    """ data = {"metric_name": {"epoch/step": [], "value": []}} """
    # draw loss curve
    fig = plt.figure(figsize=(6, 4))
    plt.yscale('log')
    k1 = list(data.keys())[0]
    per = [k for k in data[k1].keys() if k != 'value'][0]

    for key, value in data.items():
        plt.plot(value[per], value['value'], label=key)
    plt.xlabel(per)
    plt.ylabel('value')
    value_lst = flatten_list([list(value['value'])
                             for key, value in data.items()])
    plt.ylim(np.min(value_lst),
             np.percentile(value_lst, 98))
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(log_path,
                             name),
                dpi=200)
    plt.close()


def vis_loss_curve_diff_scale(log_path: str,
                              data: Dict,
                              name='loss_curve_kl_recon.jpeg',
                              color_map='Set1',
                              offset=80):
    """ 
    draw loss curve with multiple diff scales 
    data = {"metric_name": {"epoch/step": [], "value": []}}
    """
    plt.style.use('bmh')
    plt.figure(figsize=(14, 6))

    k1 = list(data.keys())[0]
    per = [k for k in data[k1].keys() if k != 'value'][0]

    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(right=0.75)
    # can be multiple scales (keys)
    keys = list(data.keys())

    # define color map
    cmap = cm.get_cmap(color_map, len(keys))

    # define linestyles
    linestyles = ['-', '--', '-.', ':']

    # define axes
    axes = [host]
    for i in range(len(keys)):
        axes.append(host.twinx())

    # adjust positions
    for i in range(len(keys)):
        if i != 0:
            axes[i].axis['right'] = axes[i].new_fixed_axis(loc="right", offset=(offset * (i-1), 0))
            axes[i].axis['right'].toggle(all=True)

    # draw loss curve
    for i, key in enumerate(keys):
        # plots
        axes[i].set_xlabel(per)
        axes[i].set_ylabel(key, size='x-small')
        axes[i].set_yscale('log')
        if isinstance(data[key], dict):
            if per in data[key].keys():
                axes[i].plot(data[key][per],
                             data[key]['value'],
                             c=cmap(i),
                             label=key,
                             linewidth=1)
            else:
                sub_dict = data[key]
                for j, (sub_key, sub_value) in enumerate(sub_dict.items()):
                    if per in sub_key:
                        axes[i].plot(sub_value,
                                     sub_dict['value'],
                                     c=cmap(i),
                                     linestyle=linestyles[j],
                                     label=sub_key,
                                     linewidth=1)
        elif isinstance(data[key], list):  # old
            sub_dict = data[key][0]
            for j, sub_key in enumerate(sub_dict.keys()):
                axes[i].plot(sub_dict[sub_key][per],
                             sub_dict[sub_key]['value'],
                             c=cmap(i),
                             linestyle=linestyles[j],
                             label=sub_key,
                             linewidth=1)
        axes[i].axis['right'].label.set_color(cmap(i))
        axes[i].tick_params(axis='y', labelcolor=cmap(i))

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_path,
                             name),
                dpi=250)
    plt.close()


def residual_plot(y_true, y_pred, save_dir='/home/yyhhli/temp.jpeg'):
    plt.scatter(y_true - y_pred, y_true)
    plt.savefig(save_dir)
    plt.close()
    pass


def ytrue_ypred_scatter(pred_dict, save_dir='/home/yyhhli/temp.jpeg'):
    # need to get rid of empty entries first
    pred_dict = {k: v for (k, v) in pred_dict.items() if len(v)}
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    y_true = pred_dict['true']
    ax.set_aspect('equal')
    for model_name, pred in pred_dict.items():
        if model_name != "true" and model_name != "__dict":
            plt.scatter(x=y_true, y=pred, label=model_name)
            plt.xlabel("Y True")
            plt.ylabel("Y Pred")
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
    LOGGER.info(f"plotted at {save_dir}")
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
    LOGGER.info(f"visualized at {save_dir}")
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
    # need to get rid of empty entries first
    pred_dict = {k: v for (k, v) in pred_dict.items() if len(v)}
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
        plt.savefig(save_dir, dpi=200)
        plt.close()
        LOGGER.info(f"visualized at {save_dir}")
        pass


def vis_pca(data: np.ndarray,
            label: Union[None, np.ndarray, list],
            save_path: str,
            label_name='NA',
            label_numeric=False):
    """ 
    Similar to vis_tsne, can be referred to when new function added.
    """
    dim = 3
    pca = PCA(n_components=dim)
    pca_result = pca.fit_transform(data)
    LOGGER.info(f"Explained variation per principal component: \
          {pca.explained_variance_ratio_}")
    plt.figure(figsize=(5, 5))
    # case 1, no label
    if label is None:
        pca_result_df = pd.DataFrame(pca_result,
                                     columns=[f"PC{str(i+1)}" for i in range(dim)])
        sns.scatterplot(x='PC1', y='PC2',
                        data=pca_result_df,
                        legend="auto",
                        alpha=0.5)
    else:
        label = np.array(label).reshape(-1, 1)
        pca_result_df = pd.DataFrame(
            pca_result, columns=[f"PC{str(i+1)}" for i in range(3)])
        pca_result_df[label_name] = label
        if label_numeric:
            # case 2: numerical label
            cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
            sns.scatterplot(
                x="PC1", y="PC2",
                hue=label_name,
                palette=cmap,
                data=pca_result_df,
                size=label_name,
                sizes=(1, 20),
                legend="auto",
                alpha=0.5)
        else:
            # case 3: categorical label
            sns.scatterplot(
                x="PC1", y="PC2",
                hue=label_name,
                palette=sns.color_palette("hls", len(np.unique(label))),
                data=pca_result_df,
                legend="auto",
                alpha=0.5)
    plt.savefig(save_path, dpi=200)
    plt.close()
    LOGGER.info(f"Visualized at {save_path}")
    pass


def vis_tsne(data: Union[np.ndarray, pd.DataFrame],
             label: Union[None, np.ndarray, list],
             save_path: str,
             label_name='NA',
             label_numeric=False):
    timer = Timer((osp.basename(__file__), "vis_tsne"))
    timer()
    dim = 3
    tsne = TSNE(n_components=dim, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    timer("t-SNE")
    plt.figure(figsize=(5, 5))
    # case 1, no label
    if label is None:
        tsne_result_df = pd.DataFrame(tsne_results,
                                      columns=[f"tSNE{str(i+1)}" for i in range(2)])
        sns.scatterplot(x='tSNE1', y='tSNE2',
                        data=tsne_result_df,
                        alpha=0.5)
    else:
        label = np.array(label).reshape(-1, 1)
        tsne_result_df = pd.DataFrame(tsne_results,
                                      columns=[f"tSNE{str(i+1)}" for i in range(dim)])
        tsne_result_df[label_name] = label
        if label_numeric:
            # case 2: numerical label
            cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
            sns.scatterplot(
                x="tSNE1", y="tSNE2",
                hue=label_name,
                palette=cmap,
                data=tsne_result_df,
                size=label_name,
                sizes=(1, 20),
                alpha=0.5)
        else:
            # case 3: categorical label
            sns.scatterplot(
                x="tSNE1", y="tSNE2",
                hue=label_name,
                palette=sns.color_palette("hls", len(np.unique(label))),
                data=tsne_result_df,
                legend="full",
                alpha=0.5)
    plt.savefig(save_path, dpi=200)
    plt.close()
    LOGGER.info(f"visualized at {save_path}")
    pass


def vis_umap(data: Union[np.ndarray, pd.DataFrame],
             label: Union[None, np.ndarray, list],
             save_path: str,
             label_name='',
             label_numeric=False):
    sns.set_style('whitegrid')
    u = UMAP().fit_transform(data)
    fig, ax = plt.subplots()
    scatter = ax.scatter(u[:, 0], u[:, 1], alpha=0.5, c=label, s=label)
    plt.gca().set_aspect('equal', 'datalim')
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    ax.legend(handles, labels, loc="upper right", title="Sizes/1000")
    if save_path:
        plt.savefig(save_path, dpi=400)
        print(f"visualized at {save_path}")
    if show:
        plt.show()
    plt.close()
    pass


def vis_heatmap(data: np.ndarray,
                xlabel: str = None,
                ylabel: str = None,
                save_path: str = '/home/yyhhli/temp.jpeg'):
    plt.figure(figsize=(5, 5))
    sns.heatmap(data, cmap='mako')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    LOGGER.info(f"Visualized at {save_path}")
    pass


def vis_clustermap(data: Dict[str, np.ndarray],
                   xlabel: str = None,
                   ylabel: str = None,
                   task_name: str = None,
                   row_cmap: str = "Spectral",
                   save_path: str = '/home/yyhhli/clustermap.jpeg'):
    # draw heatmap with clustering using clustermap
    plt.figure(figsize=(5, 5))
    [xname, yname] = list(data.keys())
    X, Y = data[xname], data[yname]
    # creat cmap for row colors (not for the heatmap)
    row_cmap = get_cmap(Y, row_cmap)
    sns.clustermap(X, cmap="vlag", row_colors=[row_cmap[k] for k in Y])
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # add legend for row_colors
    handles = [Patch(facecolor=row_cmap[cls]) for cls in row_cmap]
    plt.legend(handles,
               row_cmap,
               title=task_name,
               bbox_to_anchor=(1, 1),
               bbox_transform=plt.gcf().transFigure,
               loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    LOGGER.info(f"Visualized at {save_path}")
    pass


def get_cmap(data: Union[np.ndarray, list], cmap: str):
    if isinstance(data, list) or isinstance(data, np.ndarray):
        udata = list(set(data))
    else:
        LOGGER.error(f"not implemented case: {type(data)}")
        raise NotImplementedError
    cmap = matplotlib.cm.get_cmap(cmap)
    step = 1 / len(udata)
    return {uvalue: cmap((i + 1/2) * step) for (i, uvalue) in enumerate(udata)}


def make_snsdata(data: Union[np.ndarray, pd.DataFrame],
                 label: Union[None, np.ndarray, list] = None,):
    """
    make sns dataframe from data and label
    Args:
        data: numpy array or pandas dataframe
        label: numpy array or list, only used if data is numpy array
    Returns:
        sns dataframe
    """
    # data: col = [model1, model2, ...]; rows = [repeat1, repeat2, ...]
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, label)
    # construct sns data
    vals, names, xs, bp_xs = [], [], [], []  # bp_xs: boxplot xs
    for i, col in enumerate(data.columns):
        vals += list(data[col])
        names += [col] * len(data[col])
        bp_xs += [i + 1] * len(data[col])
        xs += list(np.random.normal(i, 0.04, data[col].values.shape[0]))  # + 1
        # adds jitter to the data points - can be adjusted
    sns_data = pd.DataFrame(
        {'names': names, 'xs': xs, "bp_xs": bp_xs, 'vals': vals})
    return sns_data


def vis_result_boxplot(data, save_path=None, box_pairs=None,
                       xlabel=None, ylabel=None, ylim=True, figsize=(4, 5), rotation=0):
    # data: pandas dataframe cols=model_names, rows=metrics

    ##### Set style options here #####
    # boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')
    # flierprops = dict(marker='o', markersize=1,
    #                   linestyle='none')
    # whiskerprops = dict(color='#00145A')
    # capprops = dict(color='#00145A')
    medianprops = dict(linewidth=1.5, linestyle='-', )  # color='#01FBEE')
    meanprops = dict(marker='D')  # , markeredgecolor='#fb0172')

    # palette = ['#FF2709', '#09FF10', '#0030D7', '#FA70B5',
    #            "#21325E", "#EF6D6D", "#573391", "#00B8A9"]

    # construct sns data
    sns_data = make_snsdata(data)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.set_style("whitegrid")
    if ylim is True:
        ylim = (0, 1)
    if ylim:
        plt.ylim(ylim)

    sns.boxplot(x='bp_xs', y='vals', hue='names',
                dodge=False,
                data=sns_data,
                palette="Set2", ax=ax,
                medianprops=medianprops, meanprops=meanprops,)
    # boxprops=boxprops, whiskerprops=whiskerprops,
    # capprops=capprops, flierprops=flierprops,
    # medianprops=medianprops, showmeans=True, meanprops=meanprops)

    plt.xticks(np.arange(0, len(data.columns)),
               data.columns, rotation=rotation)  # + 1

    sns.scatterplot(x='xs', y='vals', hue='names',
                    data=sns_data, palette="Set2", ax=ax,
                    alpha=0.5)
    # t-test annotations
    if box_pairs is not None:
        add_stat_annotation(ax, data=sns_data, x="names", y='vals', order=None,
                            box_pairs=box_pairs,  # test
                            test='t-test_ind', text_format='star', loc='outside', verbose=2)

    # no need to show legend
    ax.legend().set_visible(False)
    # add titles
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=10)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=10)
    fig = plt.gcf()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=400)
        print(f"saved to {save_path}")
    plt.show()
    return ax


def vis_result_violin_plot(data, save_path=None, box_pairs=None,
                           xlabel=None, ylabel=None, ylim=True, figsize=(4, 5)):
    """
    Args:
        data: pandas dataframe cols=model_names, rows=metrics
        save_path: path to save the figure
        box_pairs: list of tuples, each tuple is (model1, model2)
        xlabel: xlabel
        ylabel: ylabel
        ylim: ylim
        figsize: figsize
    Returns:
        ax: matplotlib axis
    """
    # construct sns data
    sns_data = make_snsdata(data)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.set_style("whitegrid")
    if ylim is True:
        ylim = (0, 1)
    if ylim:
        plt.ylim(ylim)

    plt.xticks(np.arange(0, len(data.columns)),
               data.columns, rotation=0)  # + 1

    sns.violinplot(x='xs', y='vals', hue='names',
                   data=sns_data, palette="Set2", ax=ax,
                   alpha=0.5)

    # no need to show legend
    # ax.legend().set_visible(False)
    # add titles
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=10)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=10)
    fig = plt.gcf()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=400)
        print(f"saved to {save_path}")
    plt.show()
    return ax
