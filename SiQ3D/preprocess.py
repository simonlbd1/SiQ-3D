"""
A module including some basic functions for pre-processing images.
Author: Simon lbd
"""
import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Input
import tensorflow.keras as keras
import scipy.ndimage.measurements as snm
from PIL import Image


def _make_folder(path_i, print_=True):
    """
    Make a folder
    Parameters
    ----------
    path_i : str
         The folder path
    print_ : bool, optional
        If True, print the relative path of the created folder. Default: True
    Returns
    -------
    path_i : str
        The folder path
    """
    if not os.path.exists(path_i):
        os.makedirs(path_i)
    if print_:
        print(os.path.relpath(path_i, os.getcwd()))
    return path_i


def _get_files(folder_path):
    """
    Get paths of all files in the folder
    Parameters
    ----------
    folder_path : str
        The path of the folder containing images
    Returns
    -------
    img_path : list
        A list of the file paths in the folder
    """
    img_path = []
    for img_filename in sorted(os.listdir(folder_path)):
        img_path.append(folder_path + "/" + img_filename)
    return img_path


def load_image(folder_path, print_=True):
    """
    Load a 3D image from 2D layers (without time information)
    Parameters
    ----------
    folder_path : str
        The path of the folder containing images
    print_ : int, optional
        If True, print the shape of the loaded 3D image. Default: True
    Returns
    -------
    img_array : numpy.ndarray
        The 3D array of the loaded image
    """
    img_file_path = _get_files(folder_path)
    img = []
    for img_path in img_file_path:
        img.append(cv2.imread(img_path, -1))
    img_array = np.array(img).transpose((1, 2, 0))
    if print_:
        print("Load images with shape:", img_array.shape)
    return img_array


def read_image_sequence(vol, path, file_name, z_range):
    """
    Read a specific volume

    Parameters
    ----------
    vol: int,
        a specific volume
    path: str,
        folder path
    file_name: str,
        file name of the images
    z_range: tuple,
        range of image layers

    Returns
    ----------
    img_array: numpy.ndarray,
        numpy array of the image with shape (height, width, depth)
    """
    raw_img = []
    for z in range(z_range[0], z_range[1]):
        raw_img.append(cv2.imread(path + file_name % (vol, z), -1))
    img_array = np.array(raw_img).transpose((1, 2, 0))
    return img_array


def mean_background(image):
    """
    Get mean intensity and the background intensity of the images.
    Parameters
    ----------
    image: numpy.ndarray, the raw 3D image

    Returns
    ----------
    mean_intensity: float
        Mean intensity of the image
    background_intensity: float
        Background intensity of the image
    """
    return np.mean(image), np.median(image)


def correct_bleaching(image, mean_0, background_0):
    """
    Perform photobleaching correction.
    Parameters
    ----------
    image: numpy.ndarray
        The raw 3D image, one frame
    mean_0: float
        The mean intensity of frame one
    background_0: float
        The background intensity

    Returns
    ----------
    revised_bleach_img: numpy.ndarray
        The image after implementing photobleaching correction
    """
    mean_i = np.mean(image)
    revised_bleach_img = (mean_0 - background_0) / (mean_i - background_0) * (image - background_0) + background_0
    return revised_bleach_img


def conv3d_keras(filter_size, img3d_siz):
    """
    Generate a keras model for applying 3D convolution
    Parameters
    ----------
    filter_size : tuple
    img3d_siz : tuple
    Returns
    -------
    keras.Model
        The keras model to apply 3D convolution
    """
    inputs = Input((img3d_siz[0], img3d_siz[1], img3d_siz[2], 1))
    conv_3d = Conv3D(1, filter_size, kernel_initializer=keras.initializers.Ones(), padding='same')(inputs)
    return Model(inputs=inputs, outputs=conv_3d)


def lcn_gpu(img3d, noise_level=5, filter_size=(27, 27, 1)):
    """
    Local contrast normalization by gpu
    Parameters
    ----------
    img3d : numpy.ndarray
        The raw 3D image
    noise_level : float
        The parameter to suppress the enhancement of the background noises
    filter_size : tuple, optional
        the window size to apply the normalization along x, y, and z axis. Default: (27, 27, 1)
    Returns
    -------
    norm : numpy.ndarray
        The normalized 3D image
    """
    img3d_siz = img3d.shape
    volume = filter_size[0] * filter_size[1] * filter_size[2]
    conv3d_model = conv3d_keras(filter_size, img3d_siz)
    img3d = np.expand_dims(img3d, axis=(0, 4))
    avg = conv3d_model.predict(img3d) / volume
    diff_sqr = np.square(img3d - avg)
    std = np.sqrt(conv3d_model.predict(diff_sqr) / volume)
    norm = np.divide(img3d - avg, std + noise_level)
    return norm[0, :, :, :, 0]


def _normalize_image(image, noise_level):
    """
    Normalize an 3D image by local contrast normalization
    Parameters
    ----------
    image : numpy.ndarray
        A 3D image to be normalized
    noise_level : float
        The parameter to suppress the enhancement of the background noises
    Returns
    -------
    numpy.ndarray
        The normalized image
    """
    image_norm = image - np.median(image)
    image_norm[image_norm < 0] = 0
    return lcn_gpu(image_norm, noise_level, filter_size=(27, 27, 1))


def _normalize_label(label_img):
    """
    Transform cell/non-cell image into binary (0/1)
    Parameters
    ----------
    label_img : numpy.ndarray
        Input image of cell/non-cell regions
    Returns
    -------
    numpy.ndarray
        The binarized image
    """
    return (label_img > 0).astype(int)


def crop_subregion(label_image, raw_image, classifier_size, cell_label):
    """
    Crop a sub-image of the given labeled cell

    Parameters
    ----------
    label_image: numpy.ndarray,
        the labeled image
    raw_image: numpy.ndarray,
        the raw image
    classifier_size: tuple,
        the input size of the classifier
    cell_label: int,
        the label number

    Returns
    ---------
    sub_image: numpy.ndarray,
        the crop image
    """
    x0, y0, z0 = label_image.shape[0], label_image.shape[1], label_image.shape[2]
    x_crop, y_crop, z_crop = classifier_size[1], classifier_size[2], classifier_size[3]
    image_mask = np.zeros_like(label_image) + np.median(raw_image)
    region = np.where(label_image == cell_label)
    image_mask[region] = raw_image[region]
    x_max, x_min = np.max(region[0]), np.min(region[0])
    y_max, y_min = np.max(region[1]), np.min(region[1])
    x_cen = int((x_max + x_min) / 2)
    y_cen = int((y_max + y_min) / 2)
    x_min1 = int(x_cen - x_crop / 2) if x_cen - x_crop / 2 >= 0 else 0
    x_max1 = int(x_cen + x_crop / 2) if x_cen + x_crop / 2 < x0 else x0 - 1
    y_min1 = int(y_cen - y_crop / 2) if y_cen - y_crop / 2 >= 0 else 0
    y_max1 = int(y_cen + y_crop / 2) if y_cen + y_crop / 2 < y0 else y0 - 1
    x_min1 = x_max1 - x_crop if x_max1 + 1 == x0 else x_min1
    x_max1 = x_min1 + x_crop if x_min1 == 0 else x_max1
    y_min1 = y_max1 - y_crop if y_max1 + 1 == y0 else y_min1
    y_max1 = y_min1 + y_crop if y_min1 == 0 else y_max1

    sub_image = image_mask[x_min1:x_max1, y_min1:y_max1, 0:z_crop]
    cell_coordinate = snm.center_of_mass(label_image>0, label_image, cell_label)
    return sub_image, cell_coordinate


def save_img3ts(z_range, img, path, t, use_8_bit: bool=True):
    """
    Save a 3D image at time t as 2D image sequence
    Parameters
    ----------
    z_range : range
        The range of layers to be saved
    img : numpy.ndarray
        The 3D image to be saved
    path : str
        The path of the image files to be saved.
        It should use formatted string to indicate volume number and then layer number, e.g. "xxx_t%04d_z%04i.tif"
    t : int
        The volume number for the image to be saved
    use_8_bit: bool
        The array will be transformed to 8-bit or 16-bit before saving as image.
    """
    dtype = np.uint8 if use_8_bit else np.uint16
    for i, z in enumerate(z_range):
        img2d = (img[:, :, z]).astype(dtype)
        Image.fromarray(img2d).save(path % (t, i + 1))