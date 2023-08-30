"""
A module for segmenting cells, methods including OSTU threshold, watershed, TASCAN.
Author: Simon lbd
"""
import cv2
import numpy as np
import skimage.morphology as morphology
from scipy.ndimage import filters, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed


def watershed_tascan(image_pred, z_range, min_distance_2d, min_distance_3d, samplingrate, min_size,
                     min_touching_area=None, min_touching_ratio=None, tascan_=0):
    """
    Perform watershed and tascan on the cell regions

    Parameters
    ----------
    image_pred: numpy.ndarray,
        the predicted cell regions
    z_range: int,
        the number of the z stacks of a frame
    min_distance_2d: int,
        the minimum distance value used in 2d watershed
    min_distance_3d: int,
        the minimum distance value used in 3d watershed
    samplingrate: list,
        resolution in x, y, and z axis to calculate 3D distance
    min_size: list,
        the minimum size of a cell, if the size of a cell segmented is smaller than it, the cell will be removed
    min_touching_area: int,
        the minimum value of touching area used in tascan method
    min_touching_ratio: int,
        the minimum value of touching area ratio used in tascan method
    tascan_: bool,
        if 0, won't conduct tascan method, if 1, conduct tascan method

    Returns
    -----------
    cell_instances: numpy.ndarray,
        cell instances segmented by the watershed + tascan
    """
    seg_img_2d, _ = watershed_2d(image_pred, z_range, min_distance_2d)
    _, seg_img_3d = watershed_3d(seg_img_2d, samplingrate, min_size, min_distance_3d)
    cell_instances = seg_img_3d
    if tascan_:
        cell_instances = tascan(seg_img_3d, min_touching_area, min_touching_ratio)
    return cell_instances


def watershed_2d(image_pred, z_range=21, min_distance=7):
    """
    Segment cells in each layer of the 3D image by 2D _watershed

    Parameters
    ----------
    image_pred :
        the binary image of cell region and background (predicted by 3D U-net)
    z_range :
        number of layers
    min_distance :
        the minimum cell distance allowed in the result

    Returns
    -------
    bn_output :
        binary image (cell/bg) removing boundaries detected by _watershed
    boundary :
        image of cell boundaries
    """
    boundary = np.zeros(image_pred.shape, dtype='bool')
    for z in range(z_range):
        bn_image = image_pred[:, :, z] > 0.5
        dist = distance_transform_edt(bn_image, sampling=[1, 1])
        dist_smooth = filters.gaussian_filter(dist, 2, mode='constant')

        local_maxi = peak_local_max(dist_smooth, min_distance=min_distance, indices=False)
        markers = morphology.label(local_maxi)
        labels_ws = watershed(-dist_smooth, markers, mask=bn_image)
        labels_bd = find_boundaries(labels_ws, connectivity=2, mode='outer', background=0)

        boundary[:, :, z] = labels_bd

    bn_output = image_pred > 0.5
    bn_output[boundary == 1] = 0

    return bn_output, boundary


def watershed_3d(image_watershed2d, samplingrate, min_size, min_distance):
    """
    Segment cells by 3D _watershed

    Parameters
    ----------
    image_watershed2d :
        the binary image (cell/bg) obtained by watershed_2d
    samplingrate : list
        resolution in x, y, and z axis to calculate 3D distance
    min_size :
        minimum size of cells (unit: voxels)
    min_distance :
        the minimum cell distance allowed in the result.
    Returns
    -------
    labels_wo_bd :
        label image of cells removing boundaries (set to 0)
    labels_clear :
        label image of cells before removing boundaries
    min_size :
        min_size used in this function
    cell_num :
        cell number detected in this function
    Notes
    -----
    For peak_local_max function, exclude_border=0 is important. Without it, the function will exclude the cells
    within bottom/top layers (<=min_distance layers)
    """
    dist = distance_transform_edt(image_watershed2d, sampling=samplingrate)
    dist_smooth = filters.gaussian_filter(dist, (2, 2, 0.3), mode='constant')
    local_maxi = peak_local_max(dist_smooth, min_distance=min_distance, exclude_border=0, indices=False)
    markers = morphology.label(local_maxi)
    labels_ws = watershed(-dist_smooth, markers, mask=image_watershed2d)
    labels_clear = remove_small_objects(labels_ws, min_size=min_size, connectivity=3)

    labels_bd = find_boundaries(labels_clear, connectivity=3, mode='outer', background=0)
    labels_wo_bd = labels_clear.copy()
    labels_wo_bd[labels_bd == 1] = 0
    labels_wo_bd = remove_small_objects(labels_wo_bd, min_size=min_size, connectivity=3)
    return labels_wo_bd, labels_clear


def watershed_2d_markers(image_pred, mask, z_range=21):
    """
    Recalculate cell boundaries when cell regions are overlapping

    Parameters
    ----------
    image_pred :
        the label image of cells
    mask :
        the image of the overlapping regions (0: bg; 1: one cell; >1: multiple cells)
    z_range :
        number of layers
    Returns
    -------
    labels_ws :
        the recalculated label image
    """
    labels_ws = np.zeros(image_pred.shape, dtype='int')
    for z in range(z_range):
        bn_image = np.logical_or(image_pred[:, :, z] > 0, mask[:, :, z] > 1)
        markers = image_pred[:, :, z]
        markers[np.where(mask[:, :, z] > 1)] = 0
        dist = distance_transform_edt(mask[:, :, z] > 1, sampling=[1, 1])
        labels_ws[:, :, z] = watershed(dist, markers, mask=bn_image)

    return labels_ws


def simple_threshold(image, noise_level, filter_size=3):
    """
    Perform conventional threshold method to predict cell/non-cell regions

    Parameters
    ----------
    image : numpy.ndarray,
        raw 3D image
    noise_level : float,
        threshold value used in the threshold method
    filter_size : int,
        size of the filter used for image dilation

    Returns
    ----------
        image_pred : numpy.ndarray,
            cell/non-cell regions prediction
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image_pred = []
    for z in range(image.shape[2]):
        _, seg_img = cv2.threshold(image[:, :, z], noise_level, 255, cv2.THRESH_BINARY)
        open_img = cv2.morphologyEx(seg_img, cv2.MORPH_OPEN, open_kernel)
        dilate_img = cv2.dilate(open_img, kernel, 2)
        image_pred.append(dilate_img)
    return np.array(image_pred).transpose((1, 2, 0))


def otsu_threshold(image, filter_size=3):
    """
    Perform ostu threshold method to predict cell/non-cell regions

    Parameters
    ----------
    image : numpy.ndarray,
        raw 3D image
    filter_size : int,
        size of the filter used for image dilation

    Returns
    ----------
        image_pred : numpy.ndarray,
            cell/non-cell regions prediction
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image_pred = []
    for z in range(image.shape[2]):
        _, seg_img = cv2.threshold(image[:, :, z].astype(np.uint16), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        open_img = cv2.morphologyEx(seg_img, cv2.MORPH_OPEN, open_kernel)
        dilate_img = cv2.dilate(open_img, kernel, 2)
        image_pred.append(dilate_img)
    return np.array(image_pred).transpose((1, 2, 0))


def tascan(image_ws, min_touching_area=50, touching_ratio=0.6):
    """
    Perform TASCAN method on the pre-segmented images obtained from watershed.

    Parameters
    ----------
    image_ws : numpy.ndarray,
        pre-segmented (watershed) image
    min_touching_area : int,
        minimum touching area, if the touching area between two neighboring voxels is greater than this value, the two
        voxels will be merged into one
    touching_ratio: float,
        threshold ratio of the touching area between two neighboring voxels

    Returns
    ----------
    cell_instance: numpy.ndarray,
        segmentation results, after implementing TASCAN
    """
    tascan_cluster = ClusterSuperVox(min_touching_area=min_touching_area, min_touching_ratio=touching_ratio)
    return tascan_cluster.fit(image_ws)


def get_outlayer_of_a_3d_shape(a_3d_shape_onehot):
    """
    Generate the surface of a 3D instance.

    Parameters
    ----------
    a_3d_shape_onehot: numpy.ndarray, (width, height, depth)
        the image containing a 3D instance

    Returns
    ----------
    outlayer: numpy.ndarray,
        The surface of the 3D instance
    """
    shape = a_3d_shape_onehot.shape

    a_3d_crop_diff_x1 = a_3d_shape_onehot[0:shape[0] - 1, :, :] - a_3d_shape_onehot[1:shape[0], :, :]
    a_3d_crop_diff_x2 = -a_3d_shape_onehot[0:shape[0] - 1, :, :] + a_3d_shape_onehot[1:shape[0], :, :]
    a_3d_crop_diff_y1 = a_3d_shape_onehot[:, 0:shape[1] - 1, :] - a_3d_shape_onehot[:, 1:shape[1], :]
    a_3d_crop_diff_y2 = -a_3d_shape_onehot[:, 0:shape[1] - 1, :] + a_3d_shape_onehot[:, 1:shape[1], :]
    a_3d_crop_diff_z1 = a_3d_shape_onehot[:, :, 0:shape[2] - 1] - a_3d_shape_onehot[:, :, 1:shape[2]]
    a_3d_crop_diff_z2 = -a_3d_shape_onehot[:, :, 0:shape[2] - 1] + a_3d_shape_onehot[:, :, 1:shape[2]]

    outlayer = np.zeros(shape)
    outlayer[1:shape[0], :, :] += np.array(a_3d_crop_diff_x1 == 1, dtype=np.int8)
    outlayer[0:shape[0] - 1, :, :] += np.array(a_3d_crop_diff_x2 == 1, dtype=np.int8)
    outlayer[:, 1:shape[1], :] += np.array(a_3d_crop_diff_y1 == 1, dtype=np.int8)
    outlayer[:, 0:shape[1] - 1, :] += np.array(a_3d_crop_diff_y2 == 1, dtype=np.int8)
    outlayer[:, :, 1:shape[2]] += np.array(a_3d_crop_diff_z1 == 1, dtype=np.int8)
    outlayer[:, :, 0:shape[2] - 1] += np.array(a_3d_crop_diff_z2 == 1, dtype=np.int8)

    outlayer = np.array(outlayer > 0, dtype=np.int8)

    return outlayer


def get_crop_by_pixel_val(input_3d_img, val, boundary_extend=2, crop_another_3d_img_by_the_way=None):
    """
    Crop a sub-image of a specific label

    Parameters
    ----------
    input_3d_img: numpy.ndarray, (width, height, depth)
        the input 3d image
    val: int,
        the specific label value
    boundary_extend: int,
        the width of the boundary to be extended, default 2
    crop_another_3d_img_by_the_way: numpy.ndarray, (width, height, depth)
        another 3D image, default None

    Returns
    ----------
    crop_3d_img: numpy.ndarray,
        the crop image of the given label
    crop_another_3d_img: numpy.ndarray,
        another crop image
    """
    locs = np.where(input_3d_img == val)

    shape_of_input_3d_img = input_3d_img.shape

    min_x = np.min(locs[0])
    max_x = np.max(locs[0])
    min_y = np.min(locs[1])
    max_y = np.max(locs[1])
    min_z = np.min(locs[2])
    max_z = np.max(locs[2])

    x_s = np.clip(min_x - boundary_extend, 0, shape_of_input_3d_img[0])
    x_e = np.clip(max_x + boundary_extend + 1, 0, shape_of_input_3d_img[0])
    y_s = np.clip(min_y - boundary_extend, 0, shape_of_input_3d_img[1])
    y_e = np.clip(max_y + boundary_extend + 1, 0, shape_of_input_3d_img[1])
    z_s = np.clip(min_z - boundary_extend, 0, shape_of_input_3d_img[2])
    z_e = np.clip(max_z + boundary_extend + 1, 0, shape_of_input_3d_img[2])

    crop_3d_img = input_3d_img[x_s:x_e, y_s:y_e, z_s:z_e]
    if crop_another_3d_img_by_the_way is not None:
        assert input_3d_img.shape == crop_another_3d_img_by_the_way.shape
        crop_another_3d_img = crop_another_3d_img_by_the_way[x_s:x_e, y_s:y_e, z_s:z_e]
        return crop_3d_img, crop_another_3d_img
    else:
        return crop_3d_img


class ClusterSuperVox:
    """
    A class for conducting TASCAN algorithm.

    Parameters
    ----------
    min_touching_area: int, default 50,
        the minimum value of touching area
    min_touching_ratio: float, default 0.5,
        the minimum value of touching ratio between two neighboring voxels.
    boundary_extend: int, default 2,
        the boundary width to be extended
    """
    def __init__(self, min_touching_area=50, min_touching_ratio=0.5, boundary_extend=2):
        self.min_touching_area = min_touching_area
        self.min_touching_ratio = min_touching_ratio
        self.boundary_extend = boundary_extend

        self.UN_PROCESSED = 0
        self.LONELY_POINT = -1
        self.A_LARGE_NUM = 100000000

        self.input_3d_img = None
        self.unique_vals = None
        self.restrict_area_3d = None
        self.val_labels = None
        self.val_outlayer_area = None
        self.output_3d_img = None

    def fit(self, input_3d_img, restrict_area_3d=None):
        self.input_3d_img = input_3d_img

        if restrict_area_3d is None:
            self.restrict_area_3d = np.array(input_3d_img == 0, dtype=np.int8)
        else:
            self.restrict_area_3d = restrict_area_3d

        unique_vals, unique_val_counts = np.unique(self.input_3d_img, return_counts=True)
        unique_val_counts = unique_val_counts[unique_vals > 0]
        unique_vals = unique_vals[unique_vals > 0]
        sort_locs = np.argsort(unique_val_counts)[::-1]
        self.unique_vals = unique_vals[sort_locs]

        self.val_labels = dict()
        for unique_val in self.unique_vals:
            self.val_labels[unique_val] = self.UN_PROCESSED

        self.val_outlayer_area = dict()
        for idx, unique_val in enumerate(self.unique_vals):
            self.val_outlayer_area[unique_val] = self.get_outlayer_area(unique_val)

        for idx, current_val in enumerate(self.unique_vals):
            if self.val_labels[current_val] != self.UN_PROCESSED:
                continue
            valid_neighbor_vals = self.region_query(current_val)
            if len(valid_neighbor_vals) > 0:
                self.val_labels[current_val] = current_val
                self.grow_cluster(valid_neighbor_vals, current_val)
            else:
                self.val_labels[current_val] = self.LONELY_POINT

        return self.input_3d_img

    def get_outlayer_area(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        self.boundary_extend,
                                                                        self.restrict_area_3d)
        current_crop_img_onehot = np.array(current_crop_img == current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)

        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape

        current_crop_img_onehot_outlayer[current_restrict_area > 0] = 0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)

        return current_crop_outlayer_area

    def region_query(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        self.boundary_extend,
                                                                        self.restrict_area_3d)

        current_crop_img_onehot = np.array(current_crop_img == current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)

        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape

        current_crop_img_onehot_outlayer[current_restrict_area > 0] = 0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)

        neighbor_vals, neighbor_val_counts = np.unique(current_crop_img[current_crop_img_onehot_outlayer > 0],
                                                       return_counts=True)
        neighbor_val_counts = neighbor_val_counts[neighbor_vals > 0]
        neighbor_vals = neighbor_vals[neighbor_vals > 0]
        valid_neighbor_vals = self.neighbor_check(neighbor_vals, neighbor_val_counts, current_crop_outlayer_area)

        return valid_neighbor_vals

    def neighbor_check(self, neighbor_vals, neighbor_val_counts, current_crop_outlayer_area):
        neighbor_val_counts = neighbor_val_counts[neighbor_vals > 0]
        neighbor_vals = neighbor_vals[neighbor_vals > 0]

        valid_neighbor_vals = []

        for idx, neighbor_val in enumerate(neighbor_vals):
            if neighbor_val_counts[idx] >= self.min_touching_area or \
                    (neighbor_val_counts[idx] / current_crop_outlayer_area) >= self.min_touching_ratio or \
                    (neighbor_val_counts[idx] / self.val_outlayer_area[neighbor_val]) >= self.min_touching_ratio:
                valid_neighbor_vals.append(neighbor_val)

        double_checked_valid_neighbor_vals = []
        for valid_neighbor_val in valid_neighbor_vals:
            if self.val_labels[valid_neighbor_val] == self.UN_PROCESSED or \
                    self.val_labels[valid_neighbor_val] == self.LONELY_POINT:
                double_checked_valid_neighbor_vals.append(valid_neighbor_val)

        return np.array(double_checked_valid_neighbor_vals)

    def grow_cluster(self, valid_neighbor_vals, current_val):
        valid_neighbor_vals = valid_neighbor_vals[valid_neighbor_vals > 0]
        if len(valid_neighbor_vals) > 0:
            for idx, valid_neighbor_val in enumerate(valid_neighbor_vals):
                self.val_labels[valid_neighbor_val] = current_val
                self.input_3d_img[self.input_3d_img == valid_neighbor_val] = current_val
            new_valid_neighbor_vals = self.region_query(current_val)
            self.grow_cluster(new_valid_neighbor_vals, current_val)
        else:
            return