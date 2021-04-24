''' This file provides util functions to visualize various type of data '''
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torchvision.utils as vutils
import os
import math
import pandas as pd
from typing import Dict
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


def visImg(img_array, rng = (0, 1), temp_dir='/home/yyhhli/code/image data/temp_img.png'):
    plt.imsave(temp_dir, img_array.astype(np.float), vmin = rng[0], vmax = rng[1])
    # print("Image visualized at", temp_dir)
    pass


def vis3D(img, axis=0, slice_num=None, temp_dir='/home/yyhhli/code/image data/temp_img.png'):
    if slice_num is None:
        slice_num = int(img.shape[axis]/2)
    indices = {0:None, 1:None, 2:None}
    indices[axis] = slice_num
    # print(img[tuple(slice(indices[i]) if indices[i] is None else indices[i] for i in range(3))].shape)
    visImg(img[tuple(slice(indices[i]) if indices[i] is None else indices[i] for i in range(3))], temp_dir=temp_dir)
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
    indices = {0:None, 1:None, 2:None, 3:None, 4:None}
    indices[axis + 2] = slice_num
    img_tensor_slice = img_tensor[tuple(slice(indices[i]) if indices[i] is None else indices[i] for i in range(5))]
    vutils.save_image(img_tensor_slice.data, save_dir, normalize=True, nrow=8)


def visSitk(img, axis=2, slice_num=None, temp_dir='/home/yyhhli/code/image data/temp_img.png'):
    axis2axis = {0:1, 1:2, 2:0}
    np_img = sitk.GetArrayFromImage(img)
    new_axis = axis2axis[axis]
    vis3D(np_img, axis=new_axis, slice_num=slice_num, temp_dir=temp_dir)
    

def vis_loss_curve(log_path: str, data: Dict):
    
    # draw loss curve
    fig = plt.figure(figsize=(6,4))
    plt.yscale('log')
    for key, value in data.items():
        plt.plot(value['epoch'], value['loss'], label=key)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    loss_lst = flatten_list([list(value['loss']) for key, value in data.items()])
    plt.ylim(np.min(loss_lst), 
              np.percentile(loss_lst, 98))
    plt.legend()
    fig.savefig(os.path.join(log_path, 'loss_csv.png'))


def residual_plot(y_true, y_pred):
    plt.scatter(y_true - y_pred, y_true)
    plt.savefig('/home/yyhhli/temp.jpeg')
    pass