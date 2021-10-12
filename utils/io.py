
from functools import singledispatch
import SimpleITK as sitk
import os
import os.path as osp

import numpy as np


def _load_img(img_path, type):
    # load dicom or nrrd image series in a directory
    # as sitk file
    if type == 'dcm':
        reader = sitk.ImageSeriesReader()
        dicom = reader.GetGDCMSeriesFileNames(img_path)
        reader.SetFileNames(dicom)
    else:
        reader = sitk.ImageFileReader()
        reader.SetFileName(img_path)
    try:
        img = reader.Execute()
        return img
    except Exception as ex:
        print(ex)
        return []


def load_nrrd(img_path):
    return _load_img(img_path, 'nrrd')


def load_dcm(img_path):
    return _load_img(img_path, 'dcm')


def load_mhd(img_path):
    img = sitk.ReadImage(img_path)
    np_img = sitk.GetArrayFromImage(img)
    return np_img


@singledispatch
def save_as_nrrd(img, save_path, verbose=0):
    raise NotImplementedError


@save_as_nrrd.register(sitk.Image)
def _(img: sitk.Image, save_path: str, verbose=0):
    # check directory
    save_dir = osp.dirname(save_path)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
        # check path
    if not osp.exists(save_path):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(save_path)
        writer.Execute(img)
        if verbose:
            print('Save to ', save_path)
    elif verbose:
        print(f'file {save_path} already exists.')
    return


@save_as_nrrd.register(np.ndarray)
def _(img: np.ndarray, save_path: str, verbose=0):
    img = sitk.GetImageFromArray(img)
    save_as_nrrd(img, save_path, verbose)
    pass
