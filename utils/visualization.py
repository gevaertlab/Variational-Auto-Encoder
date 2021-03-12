''' This file provides util functions to visualize various type of data '''
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torchvision.utils as vutils
import os
import math
import pandas as pd
plt.style.use('ggplot')

def visImg(img_array, temp_dir='/home/yyhhli/code/image data/temp_img.png'):
    plt.imsave(temp_dir, img_array.astype(np.float))
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
    visTemp3D(np_img, axis=new_axis, slice_num=slice_num, temp_dir=temp_dir)
    

def visLossCurve(log_path: str, start_epoch = 10, log=False):
    log_dir = os.path.join(os.getcwd(), log_path)
    log = pd.read_csv(os.path.join(log_dir, 'metrics.csv'))
    loss = list(log['loss'])
    epoch = list(log['epoch'])
    avg_val_loss = list(log['avg_val_loss'])
    epoch_lst = []
    train_loss = []
    val_loss = []
    for i in range(len(loss)):
        if not math.isnan(epoch[i]) and not math.isnan(avg_val_loss[i]):
            epoch_lst.append(epoch[i])
            val_loss.append(avg_val_loss[i])
            train_loss.append(loss[i-1])
    # draw loss curve
    fig = plt.figure(figsize=(6,4))
    plt.yscale('log')
    plt.plot(train_loss[start_epoch:], label='train loss')
    plt.plot(val_loss[start_epoch:], label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    fig.savefig(os.path.join(log_dir, 'loss_csv.png'))
    
