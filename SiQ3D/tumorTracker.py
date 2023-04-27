"""
A module for tracking tumor cells
Author: Simon lbd
"""

import os
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mgimg
import numpy as np
from functools import reduce
from .preprocess import _make_folder, read_image_sequence, mean_background, correct_bleaching, _normalize_image, \
    crop_subregion, save_img3ts, load_image, _normalize_label
from .unet3d import unet3_prediction, _divide_img, _augmentation_generator
from .segmentation import watershed_tascan, watershed_2d_markers
from .track import pr_gls_quick, initial_matching_quick, gaussian_filter, get_subregions, \
    tracking_plot_xy, tracking_plot_zx
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.measure import label
from skimage.segmentation import relabel_sequential, find_boundaries
import scipy.ndimage.measurements as snm
from scipy.stats import trim_mean

TITLE_STYLE = {'fontsize': 16, 'verticalalignment': 'bottom'}

REP_NUM_PRGLS = 5
REP_NUM_CORRECTION = 20
BOUNDARY_XY = 6
ALPHA_BLEND = 0.5


def get_random_cmap(num, seed=1):
    """
    Generate a random cmap

    Parameters
    ----------
    num : int
        The number of colors to be generated
    seed : int
        The same value will lead to the same cmap
    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The generated cmap
    """
    vals = np.linspace(0, 1, num + 1)
    np.random.seed(seed)
    np.random.shuffle(vals)
    vals = np.concatenate(([0], vals[1:]))
    cmap = plt.cm.colors.ListedColormap(plt.cm.rainbow(vals))
    cmap.colors[0, :3] = 0
    return cmap


class Paths:
    """
    Paths for storing data and results used by Tracker instance

    Attributes
    ----------
    folder_path: str,
        the path of the folder to store all data and results
    models: str,
        the path of the folder to store models
    raw_image: str,
        the path of the folder to store raw images
    unet_weights: str,
        the path of the folder to store the retrained weights of 3D U-Net
    auto_vol1: str,
        the path of the folder to store the auto-segmentation of volume 1
    manual_vol1: str,
        the path of the folder to store manual segmentation of volume 1
    track_results: str,
        the path of the folder to store tracked images
    tracking_information: str,
        the path of the folder to store the coordinates of the tracked cell
    anim: str,
        the path of the folder to store the animation of tracking in each volume
    segmentation_results: str,
        the path of the folder to store the segmentation results
    image_name: str,
        the file name of the raw image file
    unet_model_file: str,
        the file name of the pretrained 3D U-Net model
    ffn_model_file: str,
        the file name of the pretrained ffn model
    classifier_file: str,
        the file name of the pretrained classifier
    """
    def __init__(self, folder_path, image_name, unet_model_file, ffn_model_file, classifier_file):
        self.folder = folder_path
        self.models = None
        self.raw_image = None
        self.unet_weights = None
        self.auto_vol1 = None
        self.manual_vol1 = None
        self.track_results = None
        self.tracking_information = None
        self.anim = None
        self.segmentation_results = None
        self.image_name = image_name
        self.unet_model_file = unet_model_file
        self.ffn_model_file = ffn_model_file
        self.classifier_file = classifier_file

    def make_folders(self):
        """
        Make folders for storing data, models and results
        """
        print("Following folders were made under: ", os.getcwd())
        self.models = _make_folder(os.path.join(self.folder, "models/"))
        self.raw_image = _make_folder(os.path.join(self.folder, "data/"))
        self.unet_weights = _make_folder(os.path.join(self.models, "unet_weights/"))
        self.auto_vol1 = _make_folder(os.path.join(self.folder, "auto_vol1/"))
        self.manual_vol1 = _make_folder(os.path.join(self.folder, "manual_vol1/"))
        self.track_results = _make_folder(os.path.join(self.folder, "track_results/"))
        self.tracking_information = _make_folder(os.path.join(self.folder, "tracking_information/"))
        self.anim = _make_folder(os.path.join(self.folder, "anim/"))
        self.segmentation_results = _make_folder(os.path.join(self.folder, "segmentation_results/"))


class SegResults:
    """
    A class for storing the segmentation result (one frame).

    Attributes
    ----------
    image_gcn: numpy.ndarray,
        the normalized image
    cell_bg: numpy.ndarray,
        the cell/non-cell predictions by 3D U-Net
    auto_segmentation: numpy.ndarray,
        the image of individual cell instance segmented by 3D U-Net + watershed
    l_center_coordinates: list of tuple,
        the center coordinates of individual cells, unit: voxels
    r_center_coordinates: numpy.ndarray,
        transformed from l_center_coordinates, corrected by using z_xy_ratio
    """
    def __init__(self):
        self.image_gcn = None
        self.cell_bg = None
        self.auto_segmentation = None
        self.l_center_coordinates = None
        self.r_center_coordinates = None
        self.raw_image = None

    def update_seg_results(self, raw_image, image_gcn, cell_bg, auto_segmentation, l_center_coordinates,
                           r_center_coordinates):
        """
        Update the segmentation results
        """
        self.raw_image = raw_image
        self.image_gcn = image_gcn
        self.cell_bg = cell_bg
        self.auto_segmentation = auto_segmentation
        self.l_center_coordinates = l_center_coordinates
        self.r_center_coordinates = r_center_coordinates


class Segmentation:
    """
    A class for segmenting cell instances
    """
    def __init__(self, volume_num, xyz_size, z_xy_ratio, z_scaling, shrink, phenotyping):
        self.volume_num = volume_num
        self.x_size = xyz_size[0]
        self.y_size = xyz_size[1]
        self.z_size = xyz_size[2]
        self.z_xy_ratio = z_xy_ratio
        self.z_scaling = z_scaling
        self.shrink = shrink
        self.phenotyping = phenotyping
        self.noise_level = None
        self.min_size = None
        self.min_distance = None
        self.paths = None
        self.unet_model = None
        self.ffn_model = None
        self.classifier = None
        self.classifier_size = None
        self.mean_0 = None
        self.background_0 = None
        self.vol = None
        self.r_coordinates_segment_t0 = None
        self.cell_phenotypes_t0 = None
        self.segresult = SegResults()

    def set_segmentation(self, noise_level=None, min_size=None, min_distance=None):
        self.noise_level = noise_level
        self.min_size = min_size
        self.min_distance = min_distance
        print(f"Set segmentation parameters: noise_level={self.noise_level}, min_size={self.min_size}, "
              f"min_distance={self.min_distance}")

    def load_unet_ffn_classifier(self):
        """
        Load the pre-trained 3D U-Net model
        """
        self.unet_model = load_model(os.path.join(self.paths.models, self.paths.unet_model_file))
        self.unet_model.save_weights(os.path.join(self.paths.unet_weights, "weights_initial.h5"))
        self.ffn_model = load_model(os.path.join(self.paths.models, self.paths.ffn_model_file))
        if self.phenotyping:
            self.classifier = load_model(os.path.join(self.paths.models, self.paths.classifier_file))
            self.classifier_size = self.classifier.input_shape
        print("Loaded the 3D U-Net model, FFN model and classifier.")

    def _segment(self, vol):
        """
        Segment individual cells from one volume of 3D image

        Parameters
        ----------
        vol: int,
            the specific volume

        Returns
        ----------
        image_gcn: numpy.ndarray,
            the normalized image
        cell_bg: numpy.ndarray,
            the cell/non-cell predictions by 3D U-Net
        auto_segmentation: numpy.ndarray,
            the image of individual cell instance segmented by 3D U-Net + watershed
        l_center_coordinates: list of tuple,
            the center coordinates of individual cells, unit: voxels
        r_center_coordinates: numpy.ndarray,
            transformed from l_center_coordinates, corrected by using z_xy_ratio
        """
        if vol == 1:
            raw_img1 = read_image_sequence(1, self.paths.raw_image, self.paths.image_name, (1, self.z_size+1))
            self.mean_0, self.background_0 = mean_background(raw_img1)

        raw_img = read_image_sequence(vol, self.paths.raw_image, self.paths.image_name, (1, self.z_size+1))
        image_gcn = (raw_img.copy() / 65536.0)

        # photo-bleaching correction, local contrast normalization
        correct_bleach_img = correct_bleaching(raw_img, self.mean_0, self.background_0)
        norm_img = np.expand_dims(_normalize_image(correct_bleach_img, self.noise_level), axis=(0, 4))

        cell_bg = unet3_prediction(norm_img, self.unet_model, shrink=self.shrink)[0, :, :, :, 0]
        if np.max(cell_bg) <= 0.5:
            raise ValueError("No cell was detected by 3D U-Net! Please try to reduce the noise_level.")

        watershed_seg = watershed_tascan(cell_bg, self.z_size, min_distance_2d=3, min_distance_3d=self.min_distance,
                                         samplingrate=[1, 1, self.z_xy_ratio], min_size=self.min_size, tascan_=0)

        if self.phenotyping:
            for ll in np.unique(watershed_seg):
                if ll > 0:
                    sub_label_img, _ = crop_subregion(watershed_seg, raw_img, self.classifier_size, ll)
                    sub_label_img = np.expand_dims(sub_label_img, axis=(0, 4))
                    label_phenotype = self.classifier.predict(sub_label_img)
                    if label_phenotype == 1:
                        watershed_seg[watershed_seg == ll] = 0

        auto_segmentation, fw, inv = relabel_sequential(watershed_seg)

        l_center_coordinates = snm.center_of_mass(auto_segmentation > 0, auto_segmentation,
                                                  range(1, auto_segmentation.max() + 1))
        r_center_coordinates = self._transform_layer_to_real(l_center_coordinates)

        return raw_img, image_gcn, cell_bg, auto_segmentation, l_center_coordinates, r_center_coordinates

    def segment_vol1(self):
        """
        Segment the first volume
        """
        if self.unet_model is None or self.ffn_model is None:
            self.load_unet_ffn_classifier()

        self.vol = 1
        self.segresult.update_seg_results(*self._segment(vol=self.vol))
        self.r_coordinates_segment_t0 = self.segresult.r_center_coordinates.copy()
        use_8_bit = True if self.segresult.auto_segmentation.max() <= 255 else False
        save_img3ts(range(self.z_size), self.segresult.auto_segmentation,
                    self.paths.segmentation_results + "auto_segment_t%04i_z%04i.tif", self.vol, use_8_bit)

        save_img3ts(range(self.z_size), self.segresult.auto_segmentation,
                    self.paths.auto_vol1 + "auto_t%04i_z%04i.tif", 1, use_8_bit)
        print(f"Segmented volume 1 and saved it")

    def segment_one_vol(self, vol):
        """
        Segment a specific volume

        Parameters
        ----------
        vol: int,
            the specific volume
        """
        self.vol = vol
        self.segresult.update_seg_results(*self._segment(vol=self.vol))
        print(f"Finished segmenting vol {self.vol}")

    @staticmethod
    def _transform_disps(disp, factor):
        """
        Transform the coordinates with different units along z
        """
        new_disp = np.array(disp).copy()
        new_disp[:, 2] = new_disp[:, 2] * factor
        return new_disp

    def _transform_layer_to_real(self, voxel_disp):
        """
        Transform the coordinates from layer to real
        """
        return self._transform_disps(voxel_disp, self.z_xy_ratio)

    def _transform_real_to_layer(self, r_disp):
        """
        Transform the coordinates from real to layer
        """
        return np.rint(self._transform_disps(r_disp, 1 / self.z_xy_ratio)).astype(int)

    def _transform_real_to_interpolated(self, r_disp):
        """
        Transform the coordinates from real to interpolated
        """
        return np.rint(self._transform_disps(r_disp, self.z_scaling / self.z_xy_ratio)).astype(int)

    def _transform_interpolated_to_layer(self, r_disp):
        """
        Transform the coordinates from real to layer
        """
        return np.rint(self._transform_disps(r_disp, 1 / self.z_scaling)).astype(int)


class Draw:
    """
    Class for drawing figures of segmentation and tracking results
    """
    def __init__(self):
        self.x_size = None
        self.y_size = None
        self.z_size = None
        self.z_scaling = None
        self.z_xy_ratio = None
        self.segresult = None
        self.vol = None
        self.r_coordinates_tracked_t0 = None
        self.cell_num_t0 = None
        self.tracked_labels = None

    def draw_segresult(self, percentile_high=99.9):
        """
        Draw the raw image and the segmentation result of one volume by max-projection
        """
        axs, figs = self._subplots_2()
        cell_num = np.max(self.segresult.auto_segmentation)
        axs[0].set_title(f"Raw image at vol {self.vol}", fontdict=TITLE_STYLE)
        axs[1].set_title(f"Auto-segmentation at vol {self.vol}", fontdict=TITLE_STYLE)
        vmin = np.percentile(self.segresult.image_gcn, 10)
        vmax = np.percentile(self.segresult.image_gcn, percentile_high)
        axs[0].imshow(np.max(self.segresult.image_gcn, axis=2), vmin=vmin, vmax=vmax, cmap="gray")
        axs[1].imshow(np.max(self.segresult.auto_segmentation, axis=2), cmap=get_random_cmap(cell_num))

    def draw_manual_seg1(self):
        """
        Draw the cell regions and the interpolated/smoothed manual segmentation with max-projection in volume 1
        """
        axs, figs = self._subplots_2()
        axs[0].imshow(np.max(self.segresult.cell_bg, axis=2) > 0.5, cmap="gray")
        axs[0].set_title(f"Cell regions at vol {self.vol} by U-Net", fontdict=TITLE_STYLE)
        axs[1].imshow(np.max(self.seg_cells_interpolated_corrected, axis=2),
                      cmap=get_random_cmap(num=self.cell_num_t0))
        axs[1].set_title(f"Manual _segment at vol 1", fontdict=TITLE_STYLE)

    def _subplots_ffnprgls_animation(self):
        """Generate a figure to draw the FFN + PR-GLS transformation"""
        ax, fig = self._subplots_2()
        ax[0].set_title("Matching by FFN + PR-GLS (y-x plane)", fontdict=TITLE_STYLE)
        ax[1].set_title("Matching by FFN + PR-GLS (y-z plane)", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.close(fig)
        return ax, fig

    def _draw_transformation(self, ax, r_coordinates_predicted_pre, r_coordinates_segmented_post,
                             r_coordinates_predicted_post, layercoord, draw_point=True):
        """
        Draw each iteration of the tracking by FFN + PR-GLS
        """
        element1 = tracking_plot_xy(
            ax[0], r_coordinates_predicted_pre, r_coordinates_segmented_post, r_coordinates_predicted_post,
            (self.y_size, self.x_size), draw_point, layercoord)
        element2 = tracking_plot_zx(
            ax[1], r_coordinates_predicted_pre, r_coordinates_segmented_post, r_coordinates_predicted_post,
            (self.y_size, self.z_size), draw_point, layercoord)
        if layercoord:
            ax[0].set_aspect('equal', 'box')
            ax[1].set_aspect('equal', 'box')
        return element1 + element2

    def draw_correction(self, i_disp_from_vol1_updated, r_coor_predicted):
        """
        Draw the accurate correction superimposed on the cell regions

        Parameters
        ----------
        i_disp_from_vol1_updated : numpy.ndarray
            The current displacement of each cell from volume 1. Interpolated coordinates.
        r_coor_predicted : numpy.ndarray
            The current coordinates of each cell. Real coordinates
        """
        ax, fig = self._subplots_2()
        ax[0].set_title("Accurate Correction (y-x plane)", size=16)
        ax[1].set_title("Accurate Correction (y-z plane)", size=16)
        self._draw_correction(ax, r_coor_predicted, i_disp_from_vol1_updated)
        return None

    def _draw_correction(self, ax, r_coor_predicted, i_disp_from_vol1_updated):
        """
        Draw the accurate correction of cell positions after FFN + PR-GLS transformation
        """
        _ = self._draw_transformation(
            [ax[0], ax[1]],
            self._transform_real_to_layer(r_coor_predicted),
            self._transform_real_to_layer(self.segresult.r_center_coordinates),
            self._transform_real_to_layer(self.r_coordinates_tracked_t0) +
            self._transform_interpolated_to_layer(i_disp_from_vol1_updated),
            layercoord=True, draw_point=False)
        ax[0].imshow(np.max(self.segresult.cell_bg[:, :, :], axis=2) > 0.5, cmap="gray",
                     extent=(0, self.y_size - 1, self.x_size - 1, 0))
        ax[1].imshow(np.max(self.segresult.cell_bg[:, :, :], axis=0).T > 0.5, aspect=self.z_xy_ratio,
                     cmap="gray", extent=(0, self.y_size - 1, self.z_size - 1, 0))
        return None

    def draw_overlapping(self, cells_on_boundary_local, volume2, i_disp_from_vol1_updated):
        """
        Draw the overlapping of cell regions (gray) and the labels before/after matching

        Parameters
        ----------
        cells_on_boundary_local : numpy.ndarray
            A 1d array of cells on boundary (1, ignored in tracking) or not (0).
        volume2 : int
            The current volume
        i_disp_from_vol1_updated : numpy.ndarray
            The displacement of each cell from volume 1. Interpolated coordinates.
        """
        self.tracked_labels, _ = self._transform_motion_to_image(cells_on_boundary_local, i_disp_from_vol1_updated)
        self._draw_matching(volume2)
        plt.pause(0.1)
        return None

    def _draw_matching(self, volume2):
        """
        Draw the overlapping of cell and labels
        """
        axs, figs = self._subplots_4()
        self._draw_before_matching(axs[0], axs[1], volume2)
        self._draw_after_matching(axs[2], axs[3], volume2)
        plt.tight_layout()
        return None

    def _draw_before_matching(self, ax1, ax2, volume2):
        """Draw overlapping of cells and labels before matching"""
        ax1.imshow(np.max(self.segresult.cell_bg[:, :, :], axis=2) > 0.5, cmap="gray")
        ax1.imshow(np.max(self.seg_cells_interpolated_corrected[:, :, self.Z_RANGE_INTERP], axis=2),
                   cmap=get_random_cmap(num=self.cell_num_t0), alpha=ALPHA_BLEND)

        ax2.imshow(np.max(self.segresult.cell_bg[:, :, :], axis=0).T > 0.5, aspect=self.z_xy_ratio,
                   cmap="gray")
        ax2.imshow(np.max(self.seg_cells_interpolated_corrected[:, :, self.Z_RANGE_INTERP], axis=0).T,
                   cmap=get_random_cmap(num=self.cell_num_t0), aspect=self.z_xy_ratio, alpha=ALPHA_BLEND)
        ax1.set_title(f"Before matching: Cells at vol {volume2} + Labels at vol {self.vol} (y-x plane)",
                      fontdict=TITLE_STYLE)
        ax2.set_title(f"Before matching (y-z plane)",
                      fontdict=TITLE_STYLE)

    def _draw_after_matching(self, ax1, ax2, volume2, legend=True):
        """Draw overlapping of cells and labels after matching"""
        ax1.imshow(np.max(self.segresult.cell_bg[:, :, :], axis=2) > 0.5, cmap="gray")
        ax1.imshow(np.max(self.tracked_labels, axis=2),
                   cmap=get_random_cmap(num=len(np.unique(self.tracked_labels)) - 1), alpha=ALPHA_BLEND)

        ax2.imshow(np.max(self.segresult.cell_bg[:, :, :], axis=0).T > 0.5, aspect=self.z_xy_ratio,
                   cmap="gray")
        ax2.imshow(np.max(self.tracked_labels, axis=0).T,
                   cmap=get_random_cmap(num=len(np.unique(self.tracked_labels))-1),
                   aspect=self.z_xy_ratio, alpha=ALPHA_BLEND)
        if legend:
            ax1.set_title(f"After matching: Cells at vol {volume2} + Labels at vol {volume2} (y-x plane)",
                          fontdict=TITLE_STYLE)
            ax2.set_title(f"After matching (y-z plane)",
                          fontdict=TITLE_STYLE)
        return None

    def subplots_tracking(self):
        """
        Generate a (3, 2) layout subplots in a figure
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure to draw tracking process
        ax : list of matplotlib.axes.Axes
            The subplots to show each panel
        """
        fig, axs = plt.subplots(3, 2, figsize=(14, int(21 * self.x_size / self.y_size)))
        ax = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]
        return fig, ax

    def _draw_matching_6panel(self, target_volume, ax, r_coor_predicted_mean, i_disp_from_vol1_updated):
        """
        Draw the tracking process in a specific volume
        """
        for ax_i in ax:
            ax_i.cla()
        plt.suptitle(f"Tracking results at vol {target_volume}", size=16)

        _ = self._draw_transformation([ax[0], ax[1]], self.history.r_tracked_coordinates[target_volume - 2],
                                      self.segresult.r_center_coordinates, r_coor_predicted_mean, layercoord=False)
        self._draw_correction([ax[2], ax[3]], r_coor_predicted_mean, i_disp_from_vol1_updated)
        self._draw_after_matching(ax[4], ax[5], target_volume, legend=False)
        self._set_layout_anim()
        for axi in ax:
            plt.setp(axi.get_xticklabels(), visible=False)
            plt.setp(axi.get_yticklabels(), visible=False)
            axi.tick_params(axis='both', which='both', length=0)
            axi.axis("off")
        return None

    @staticmethod
    def _set_layout_anim():
        """set the layout to show the tracking process dynamically in the notebook backend"""
        plt.tight_layout()
        plt.subplots_adjust(right=0.9, bottom=0.1)

    def _subplots_2(self):
        """
        Generate 2 subplots in a figure
        """
        fig, ax = plt.subplots(1, 2, figsize=(20, int(12 * self.x_size / self.y_size)))
        plt.tight_layout()
        return ax, fig

    def _subplots_3(self):
        """
        Generate 3 subplots in a figure
        """
        fig, ax = plt.figure(figsize=(20, int(24 * self.x_size / self.z_size)))
        ax = plt.subplot(221), plt.subplot(222), plt.subplot(223)
        plt.tight_layout()
        return ax, fig

    def _subplots_4(self):
        """
        Generate 4 subplots in a figure
        """
        fig, axs = plt.subplots(2, 2, figsize=(20, int(24 * self.x_size / self.y_size)))
        ax = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
        plt.tight_layout()
        return ax, fig

    def _transform_interpolated_to_layer(self, i_disp_from_vol1_updated):
        """Overridden in Tracker"""
        raise NotImplementedError("Must override this method")

    def _transform_motion_to_image(self, cells_on_boundary_local, i_disp_from_vol1_updated):
        """Overridden in Tracker"""
        raise NotImplementedError("Must override this method")

    def _transform_real_to_layer(self, r_coor_predicted):
        """Overridden in Tracker"""
        raise NotImplementedError("Must override this method")


class TrainUnet:
    """
    A class for training 3d U-Net
    """
    def __init__(self):
        self.x_size = None
        self.y_size = None
        self.z_size = None
        self.paths = None
        self.mean_0 = None
        self.noise_level = None
        self.background_0 = None
        self.manual_segmentation_relabels = None
        self.raw_image_vol1 = None
        self.train_norm_image = None
        self.label_vol1 = None
        self.train_norm_label = None
        self.train_subimage = None
        self.train_subcells = None
        self.unet_model = None
        self.train_generator = None
        self.valid_data = None
        self.val_losses = None

    def _remove_2d_boundary(self, labels3d):
        """
        Remove boundaries between touching cells in x-y plane

        Parameters
        ----------
        labels3d: numpy.ndarray,
            the 3D image of cell labels

        Returns
        ----------
        new_labels: numpy.ndarray,
            the new label image with the boundaries removed
        """
        new_labels = labels3d.copy()
        for z in range(self.z_size):
            labels = new_labels[:, :, z]
            labels[find_boundaries(labels, mode='outer') == 1] = 0
        return new_labels

    def _retrain_preprocess(self):
        """
        Data preprocessing for retraining the 3D U-Net model
        """
        self.raw_image_vol1 = read_image_sequence(1, self.paths.raw_image, self.paths.image_name, (1, 1 + self.z_size))
        corrected_bleach_image = correct_bleaching(self.raw_image_vol1, self.mean_0, self.background_0)
        self.train_norm_image = _normalize_image(corrected_bleach_image, self.noise_level)
        self.label_vol1 = self._remove_2d_boundary(self.manual_segmentation_relabels) > 0
        self.train_norm_label = _normalize_label(self.label_vol1)
        print("Images were normalized")

        self.train_subimage = _divide_img(self.train_norm_image, self.unet_model.input_shape[1:4])
        self.train_subcells = _divide_img(self.train_norm_label, self.unet_model.input_shape[1:4])
        print("Images were divided")

        image_gen = ImageDataGenerator(rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, horizontal_flip=True, fill_mode='reflect')

        self.train_generator = _augmentation_generator(self.train_subimage, self.train_subcells, image_gen,
                                                       batch_siz=8)
        self.valid_data = (self.train_subimage, self.train_subcells)
        print("Data for training 3D U-Net were prepared")

    def retrain_unet(self, iteration=10, weights_name="unet_weights_retrain_"):
        """
        Retrain the 3D U-Net by using the manually corrected segmentation in volume 1.

        Parameters
        ----------
        iteration: int,
            the number of epochs to train the 3D U-Net model, default: 10
        weights_name: str,
            the filename of the unet weights to be saved.
        """
        self._retrain_preprocess()

        self.unet_model.compile(loss='binary_crossentropy', optimizer="adam")
        self.unet_model.load_weights(os.path.join(self.paths.unet_weights, 'weights_initial.h5'))

        val_loss = self.unet_model.evaluate(self.train_subimage, self.train_subcells)
        print("Val_loss before retraining: ", val_loss)
        self.val_losses = [val_loss]
        self._draw_retrain(step=0)

        for step in range(1, iteration + 1):
            self.unet_model.fit_generator(self.train_generator, validation_data=self.valid_data, epochs=1,
                                          steps_per_epoch=60)
            loss = self.unet_model.history.history["val_loss"][-1]
            if loss < min(self.val_losses):
                print("val_loss updated from ", min(self.val_losses), " to ", loss)
                self.unet_model.save_weights(os.path.join(self.paths.unet_weights, weights_name + f"epoch{step}.h5"))
                self._draw_retrain(step)
            self.val_losses.append(loss)

    def select_unet_weights(self, epoch, weights_name="unet_weights_retrain_"):
        """
        Select a satisfied unet weights
        """
        if epoch == 0:
            self.unet_model.load_weights(os.path.join(self.paths.unet_weights, 'weights_initial.h5'))
        else:
            self.unet_model.load_weights((os.path.join(self.paths.unet_weights, weights_name + f"epoch{epoch}.h5")))
            self.unet_model.save(os.path.join(self.paths.unet_weights, "unet3_retrained.h5"))

    def _draw_retrain(self, step):
        """
        Draw the ground truth and the updated predictions during retraining the unet
        """
        train_prediction = np.squeeze(unet3_prediction(np.expand_dims(self.train_norm_image, axis=(0, 4)),
                                                       self.unet_model))
        fig, axs = plt.subplots(1, 2, figsize=(20, int(12 * self.x_size / self.y_size)))
        axs[0].imshow(np.max(self.label_vol1, axis=2), cmap="gray")
        axs[1].imshow(np.max(train_prediction, axis=2) > 0.5, cmap="gray")
        axs[0].set_title("Cell regions from manual segmentation at vol 1", fontdict=TITLE_STYLE)
        axs[1].set_title(f"Cell prediction at epoch {step} at vol 1", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.pause(0.1)


class History:
    """
    A class for storing the displacements, coordinates, and tracking animations

    Attributes
    ----------
    r_displacements: list of numpy.ndarray,
        the displacements of each cell in each volume relative to their positions in volume 1. (cell_num, 3)
    r_segmented_coordinates: list of numpy.ndarray,
        the positions of each cell segmented by 3D U-Net + watershed in each volume.
    r_tracked_coordinates: list of numpy.ndarray,
        the positions of each cell tracked by FFN + PR-GLS + corrections in each volume.
    anim: list,
        the images of tracking process in each volume (from volume 2)
    """
    def __init__(self):
        self.r_displacements = []
        self.r_segmented_coordinates = []
        self.r_tracked_coordinates = []
        self.cell_phenotypes = []
        self.aim = []


class Tracker(Segmentation, Draw, TrainUnet):
    def __init__(self, volume_num, xyz_size, z_xy_ratio, noise_level, min_size, min_distance, beta_tk,
                 lambda_tk, maxiter_tk, folder_path, image_name, unet_model_file, ffn_model_file, classifier_file,
                 shrink=(24, 24, 2), phenotyping=True):
        z_scaling = int(z_xy_ratio * 2)
        Segmentation.__init__(self, volume_num, xyz_size, z_xy_ratio, z_scaling, shrink, phenotyping)
        self.paths = Paths(folder_path, image_name, unet_model_file, ffn_model_file, classifier_file)
        self.history = History()
        self.noise_level = noise_level
        self.min_size = min_size
        self.min_distance = min_distance
        self.beta_tk = beta_tk
        self.lambda_tk = lambda_tk
        self.maxiter_tk = maxiter_tk
        self.phenotyping = phenotyping
        self.use_8_bit = True
        self.seg_cells_interpolated_corrected = None
        self.Z_RANGE_INTERP = None
        self.cell_num_t0 = None
        self.region_list = None
        self.region_width = None
        self.region_xyz_min = None
        self.pad_x = None
        self.pad_y = None
        self.pad_z = None
        self.ffn_model = None
        self.classifier = None
        self.label_padding = None
        self.cells_on_boundary = None
        self.paths.make_folders()

    def set_tracking(self, beta_tk, lambda_tk, maxiter_tk):
        """
        Set tracking parameters
        """
        self.beta_tk = beta_tk
        self.lambda_tk = lambda_tk
        self.maxiter_tk = maxiter_tk
        print(f"Set tracking parameters as: beta_tk={self.beta_tk}, lambda_tk={self.lambda_tk},"
              f"maxiter_tk={self.maxiter_tk}")

    def load_manual_seg(self):
        """
        Load the manually corrected segmentation in the "/manual_vol1" folder
        """
        manual_segmentation = load_image(self.paths.manual_vol1, print_=False)
        print("Loaded manual segmentation at vol 1.")
        self.manual_segmentation_relabels, _, _ = relabel_sequential(manual_segmentation)
        if self.manual_segmentation_relabels.max() > 255:
            self.use_8_bit = False

    def load_ffn_classifier(self):
        """
        Load the pre-trained FFN model and cell phenotypes classifier
        """
        self.ffn_model = load_model(os.path.join(self.paths.models, self.paths.ffn_model_file))
        if self.phenotyping:
            self.classifier = load_model(os.path.join(self.paths.models, self.paths.classifier_file))
        print("Loaded the FFN model and cell phenotypes classifier")

    def interpolate_seg(self):
        """
        Interpolate the images along z axis in volume 1 and save the results in "track_results" folder
        """
        self.seg_cells_interpolated_corrected = self._interpolate()
        self.Z_RANGE_INTERP = range(self.z_scaling // 2, self.seg_cells_interpolated_corrected.shape[2],
                                    self.z_scaling)

        # re-segmentation
        self.seg_cells_interpolated_corrected = self._relabel_separated_cells(self.seg_cells_interpolated_corrected)
        self.manual_segmentation_relabels = self.seg_cells_interpolated_corrected[:, :, self.Z_RANGE_INTERP]

        # save labels in the first volume (interpolated)
        save_img3ts(range(0, self.z_size), self.manual_segmentation_relabels,
                    self.paths.track_results + "track_results_t%04i_z%04i.tif", t=1, use_8_bit=self.use_8_bit)

        # calculate coordinates of cell centers at t=1
        center_points_t0 = snm.center_of_mass(self.manual_segmentation_relabels > 0,
                                              self.manual_segmentation_relabels,
                                              range(1, self.manual_segmentation_relabels.max() + 1))
        r_coordinates_manual_vol1 = self._transform_layer_to_real(center_points_t0)
        self.r_coordinates_tracked_t0 = r_coordinates_manual_vol1.copy()
        self.cell_num_t0 = r_coordinates_manual_vol1.shape[0]

    def _interpolate(self):
        """Interpolate/smoothen a 3D image"""
        seg_cells_interpolated, seg_cell_or_bg = gaussian_filter(
            self.manual_segmentation_relabels, z_scaling=self.z_scaling, smooth_sigma=2.5)
        seg_cells_interpolated_corrected = watershed_2d_markers(
            seg_cells_interpolated, seg_cell_or_bg, z_range=self.z_size * self.z_scaling + 10)
        return seg_cells_interpolated_corrected[5:self.x_size + 5, 5:self.y_size + 5,
                                                5:self.z_size * self.z_scaling + 5]

    @staticmethod
    def _relabel_separated_cells(seg_cells_interpolated):
        """
        Relabel the separate cells that were incorrectly labeled as the same one
        """
        num_cells = np.size(np.unique(seg_cells_interpolated)) - 1
        seg_cells_interpolated_corrected = label(seg_cells_interpolated, connectivity=3)
        if num_cells != np.max(seg_cells_interpolated_corrected):
            print(f"WARNING: {num_cells} cells were manually labeled while the program found "
                  f"{np.max(seg_cells_interpolated_corrected)} separated cells and corrected it")
        return seg_cells_interpolated_corrected

    def cal_subregions(self):
        """
        Calculate the subregions of cells and the padded images to accelerate the accurate correction in tracking.
        """
        seg_16 = self.seg_cells_interpolated_corrected.astype("int16")

        self.region_list, self.region_width, self.region_xyz_min = get_subregions(seg_16, seg_16.max())
        self.pad_x, self.pad_y, self.pad_z = np.max(self.region_width, axis=0)
        self.label_padding = np.pad(seg_16,
                                    pad_width=((self.pad_x, self.pad_x),
                                               (self.pad_y, self.pad_y),
                                               (self.pad_z, self.pad_z)),
                                    mode='constant') * 0

    def initiate_tracking(self):
        """
        Initiate the lists to store the displacement/coordinates histories from volume 1 (t0)
        """
        self.interpolate_seg()
        self.cal_subregions()

        self.cells_on_boundary = np.zeros(self.cell_num_t0).astype(int)
        cell_phenotypes_t0 = [0 for _ in range(self.cell_num_t0)]
        self.history.r_displacements = []
        self.history.r_displacements.append(np.zeros((self.cell_num_t0, 3)))
        self.history.r_segmented_coordinates = []
        self.history.r_segmented_coordinates.append(self.r_coordinates_segment_t0)
        self.history.r_tracked_coordinates = []
        self.history.r_tracked_coordinates.append(self.r_coordinates_tracked_t0)
        self.history.cell_phenotypes.append(cell_phenotypes_t0)
        self.history.anim = []
        print("Initiated coordinates for tracking (from vol 1)")

    def match(self, target_volume):
        """
        Match cells in volume 1 with the target_volume

        Parameters
        ----------
        target_volume : int
            The target volume to be matched

        Returns
        -------
        anim : matplotlib.animation.ArtistAnimation
            The animation including each iteration of the FFN + PR-GLS predictions
        [cells_on_boundary_local, target_volume, i_disp_from_vol1_updated, r_coor_predicted] : list
            The matching results used to draw figures
        """
        # generate automatic _segment in current volume
        self.segresult.update_seg_results(*self._segment(target_volume))

        # track by ffn + prgls
        r_coor_predicted, anim = self._predict_pos_once(source_volume=1, draw=True)

        # boundary cells
        cells_bd = self._get_cells_on_boundary(r_coor_predicted)
        cells_on_boundary_local = self.cells_on_boundary.copy()
        cells_on_boundary_local[cells_bd] = 1

        # accurate correction
        _, i_disp_from_vol1_updated = \
            self._accurate_correction(cells_on_boundary_local, r_coor_predicted)
        print(f"Matching between vol 1 and vol {target_volume} was computed")
        return anim, [cells_on_boundary_local, target_volume, i_disp_from_vol1_updated, r_coor_predicted]

    def _predict_pos_once(self, source_volume, draw=False):
        """
        Predict cell coordinates using the transformation parameters in all repetitions from fnn_prgls()
        """
        # fitting the parameters for transformation
        C_t, BETA_t, coor_intermediate_list = self._fit_ffn_prgls(
            REP_NUM_PRGLS, self.history.r_segmented_coordinates[source_volume - 1])

        # Transform the coordinates
        r_coordinates_predicted = self.history.r_tracked_coordinates[source_volume - 1].copy()

        if draw:
            ax, fig = self._subplots_ffnprgls_animation()
            plt_objs = []
            for i in range(len(C_t)):
                r_coordinates_predicted, r_coordinates_predicted_pre = self._predict_one_rep(
                    r_coordinates_predicted, coor_intermediate_list[i], BETA_t[i], C_t[i])
                plt_obj = self._draw_transformation(
                    ax, r_coordinates_predicted_pre, self.segresult.r_center_coordinates,
                    r_coordinates_predicted, layercoord=False)
                plt_objs.append(plt_obj)
            anim = animation.ArtistAnimation(fig, plt_objs, interval=200).to_jshtml()
        else:
            for i in range(len(C_t)):
                r_coordinates_predicted, r_coordinates_predicted_pre = self._predict_one_rep(
                    r_coordinates_predicted, coor_intermediate_list[i], BETA_t[i], C_t[i])
            anim = None

        return r_coordinates_predicted, anim

    def _fit_ffn_prgls(self, rep, r_coordinates_segment_pre):
        """
        Appliy FFN + PR-GLS from t1 to t2 (multiple times) to get transformation parameters to predict cell coordinates

        Parameters
        ----------
        rep : int
            The number of repetitions of (FFN + max_iteration times of PR-GLS)
        r_coordinates_segment_pre : numpy.ndarray
            Coordinates of cells in previous volume. Shape: (cell_num, 3)

        Returns
        -------
        C_t : list
            List of C in each repetition (to predict the transformed coordinates)
        BETA_t : list
            List of the parameter beta used in each repetition (to predict coordinates)
        coor_intermediate_list : list
            List of the pre-transformed coordinates of automatically segmented cells in each repetition
            (to predict coordinates)
        """
        corr_intermediate = r_coordinates_segment_pre.copy()
        C_t = []
        BETA_t = []
        coor_intermediate_list = []
        for i in range(rep):
            coor_intermediate_list.append(corr_intermediate)
            C, corr_intermediate = self._ffn_prgls_once(i, corr_intermediate)
            C_t.append(C)
            BETA_t.append(self.beta_tk * (0.8 ** i))
        return C_t, BETA_t, coor_intermediate_list

    def _ffn_prgls_once(self, i, r_coordinates_segment_pre):
        """Apply one iteration of FFN + PR-GLS"""
        init_match = initial_matching_quick(self.ffn_model, r_coordinates_segment_pre,
                                            self.segresult.r_center_coordinates, 20)
        pre_transformation_pre = r_coordinates_segment_pre.copy()
        P, r_coordinates_segment_post, C = pr_gls_quick(pre_transformation_pre,
                                                        self.segresult.r_center_coordinates,
                                                        init_match,
                                                        BETA=self.beta_tk * (0.8 ** i),
                                                        max_iteration=self.maxiter_tk,
                                                        LAMBDA=self.lambda_tk)
        return C, r_coordinates_segment_post

    def _predict_one_rep(self, r_coordinates_predicted_pre, coor_intermediate_list, BETA_t, C_t):
        """
        Predict cell coordinates using one set of the transformation parameters from fnn_prgls()

        Parameters
        ----------
        r_coordinates_predicted_pre: the coordinates before transformation
        coor_intermediate_list, BETA_t, C_t: one set of the transformation parameters

        Returns
        ----------
        r_coordinates_predicted_post: the coordinates after transformation
        """

        length_auto_segmentation = np.size(coor_intermediate_list, axis=0)

        r_coordinates_predicted_tile = np.tile(r_coordinates_predicted_pre, (length_auto_segmentation, 1, 1))
        coor_intermediate_tile = np.tile(coor_intermediate_list, (self.cell_num_t0, 1, 1)).transpose((1, 0, 2))
        Gram_matrix = np.exp(-np.sum(np.square(r_coordinates_predicted_tile - coor_intermediate_tile),
                                     axis=2) / (2 * BETA_t * BETA_t))

        r_coordinates_predicted_post = r_coordinates_predicted_pre + np.dot(C_t, Gram_matrix).T

        return r_coordinates_predicted_post, r_coordinates_predicted_pre

    def _get_cells_on_boundary(self, r_coordinates_prgls):
        """
        Get cell near the boundary of the image
        """
        boundary_xy = BOUNDARY_XY
        cells_bd = np.where(reduce(
            np.logical_or,
            [r_coordinates_prgls[:, 0] < boundary_xy,
             r_coordinates_prgls[:, 1] < boundary_xy,
             r_coordinates_prgls[:, 0] > self.x_size - boundary_xy,
             r_coordinates_prgls[:, 1] > self.y_size - boundary_xy,
             r_coordinates_prgls[:, 2] / self.z_xy_ratio < 0,
             r_coordinates_prgls[:, 2] / self.z_xy_ratio > self.z_size]))
        return cells_bd

    def _accurate_correction(self, cells_on_boundary_local, r_coor_predicted):
        """
        Correct center positions of cells based on the cell regions detected by unet and intensities in raw image
        """
        r_disp_from_vol1_updated = self.history.r_displacements[-1] + (r_coor_predicted -
                                                                       self.history.r_tracked_coordinates[-1])
        i_disp_from_vol1_updated = self._transform_real_to_interpolated(r_disp_from_vol1_updated)
        for i in range(REP_NUM_CORRECTION):
            # update positions (from vol1) by correction
            r_disp_from_vol1_updated, i_disp_from_vol1_updated, r_disp_correction = \
                self._correction_once_interp(i_disp_from_vol1_updated, cells_on_boundary_local)

            # stop the repetition if correction converged
            stop_flag = self._evaluate_correction(r_disp_correction)
            if i == REP_NUM_CORRECTION - 1 or stop_flag:
                break
        return r_disp_from_vol1_updated, i_disp_from_vol1_updated

    def _correction_once_interp(self, i_displacement_from_vol1, cell_on_bound):
        """
        Correct the tracking for once (in interpolated image)
        """
        # generate current image of labels from the manually corrected _segment in volume 1
        i_l_tracked_cells_prgls_0, i_l_overlap_prgls_0 = self._transform_cells_quick(i_displacement_from_vol1)
        l_tracked_cells_prgls = i_l_tracked_cells_prgls_0[:, :,
                                                        self.z_scaling // 2:self.z_size * self.z_scaling:self.z_scaling]
        l_overlap_prgls = i_l_overlap_prgls_0[:, :,
                                              self.z_scaling // 2:self.z_size * self.z_scaling:self.z_scaling]

        # overlapping regions of multiple cells are discarded before correction to avoid cells merging
        l_tracked_cells_prgls[np.where(l_overlap_prgls > 1)] = 0

        for i in np.where(cell_on_bound == 1)[0]:
            l_tracked_cells_prgls[l_tracked_cells_prgls == (i + 1)] = 0

        # accurate correction of displacement
        l_coordinates_prgls_int_move = \
            self.r_coordinates_tracked_t0 * np.array([1, 1, 1 / self.z_xy_ratio]) + \
            i_displacement_from_vol1 * np.array([1, 1, 1 / self.z_scaling])
        l_centers_unet_x_prgls = snm.center_of_mass(
            self.segresult.cell_bg + self.segresult.image_gcn, l_tracked_cells_prgls,
            range(1, self.seg_cells_interpolated_corrected.max() + 1))
        l_centers_unet_x_prgls = np.asarray(l_centers_unet_x_prgls)
        l_centers_prgls = np.asarray(l_coordinates_prgls_int_move)

        lost_cells = np.where(np.isnan(l_centers_unet_x_prgls)[:, 0])

        r_displacement_correction = l_centers_unet_x_prgls - l_centers_prgls
        r_displacement_correction[lost_cells, :] = 0
        r_displacement_correction[:, 2] = r_displacement_correction[:, 2] * self.z_xy_ratio

        # calculate the corrected displacement from vol #1
        r_displacement_from_vol1 = i_displacement_from_vol1 * np.array(
            [1, 1, self.z_xy_ratio / self.z_scaling]) + r_displacement_correction
        i_displacement_from_vol1_new = self._transform_real_to_interpolated(r_displacement_from_vol1)

        return r_displacement_from_vol1, i_displacement_from_vol1_new, r_displacement_correction

    def _transform_cells_quick(self, vectors3d):
        """
        Generate a image with labels indicating the moved cells.
        Parameters
        ----------
        vectors3d : numpy.ndarray
            Movements of each cell
        Returns
        -------
        output : numpy.ndarray
            The new image with moved cells
        mask : numpy.ndarray
            The new image with the
        """
        label_moved = self.label_padding.copy()
        mask = label_moved.copy()
        for ll in range(0, len(self.region_list)):
            new_x_min = self.region_xyz_min[ll][0] + vectors3d[ll, 0] + self.pad_x
            new_y_min = self.region_xyz_min[ll][1] + vectors3d[ll, 1] + self.pad_y
            new_z_min = self.region_xyz_min[ll][2] + vectors3d[ll, 2] + self.pad_z
            subregion_previous = label_moved[new_x_min:new_x_min + self.region_width[ll][0],
                                             new_y_min:new_y_min + self.region_width[ll][1],
                                             new_z_min:new_z_min + self.region_width[ll][2]]
            if subregion_previous.shape != self.region_list[ll].shape:
                continue
            subregion_new = subregion_previous * (1 - self.region_list[ll]) + self.region_list[ll] * (ll + 1)
            label_moved[new_x_min:new_x_min + self.region_width[ll][0],
                        new_y_min:new_y_min + self.region_width[ll][1],
                        new_z_min:new_z_min + self.region_width[ll][2]] = subregion_new
            mask[new_x_min:new_x_min + self.region_width[ll][0],
                 new_y_min:new_y_min + self.region_width[ll][1],
                 new_z_min:new_z_min + self.region_width[ll][2]] += (self.region_list[ll] > 0).astype("int8")
        output = label_moved[self.pad_x:-self.pad_x, self.pad_y:-self.pad_y, self.pad_z:-self.pad_z]
        mask = mask[self.pad_x:-self.pad_x, self.pad_y:-self.pad_y, self.pad_z:-self.pad_z]

        return output, mask

    def _evaluate_correction(self, r_displacement_correction):
        """
        evaluate if the accurate correction should be stopped
        """
        i_disp_test = r_displacement_correction.copy()
        i_disp_test[:, 2] *= self.z_scaling / self.z_xy_ratio
        if np.nanmax(np.abs(i_disp_test)) >= 0.5:
            return False
        else:
            return True

    def _transform_motion_to_image(self, cells_on_boundary_local, i_disp_from_vol1_updated):
        """
        Transform the predicted movements to the moved labels in 3D image
        """
        i_tracked_cells_corrected, i_overlap_corrected = self._transform_cells_quick(i_disp_from_vol1_updated)
        # re-calculate boundaries by _watershed
        i_tracked_cells_corrected[i_overlap_corrected > 1] = 0
        for i in np.where(cells_on_boundary_local == 1)[0]:
            i_tracked_cells_corrected[i_tracked_cells_corrected == (i + 1)] = 0

        tracked_labels = watershed_2d_markers(
            i_tracked_cells_corrected[:, :, self.Z_RANGE_INTERP], i_overlap_corrected[:, :, self.Z_RANGE_INTERP],
            z_range=self.z_size)

        phenotypes_pre = [0 for _ in range(self.cell_num_t0)]
        if self.phenotyping:
            for ll in np.unique(tracked_labels):
                if ll > 0:
                    sub_label_img, _ = crop_subregion(tracked_labels, self.segresult.raw_image,
                                                      self.classifier_size, ll)
                    sub_label_img = np.expand_dims(sub_label_img, axis=(0, 4))
                    label_phenotype = self.classifier.predict(sub_label_img)
                    if label_phenotype == 1:
                        tracked_labels[tracked_labels == ll] = 0
                        phenotypes_pre[ll - 1] = 1

        return tracked_labels, phenotypes_pre

    def track(self, fig, ax, from_volume=2):
        """
        Track cells from a specific volume

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to draw tracking results
        ax : matplotlib.figure.Figure
            The list of 6 subplots to draw tracking results
        from_volume : int, optional
            The volume from which to track the cells. Should be >= 2 and <= last tracked volume + 1. Default: 2
        """
        self._reset_tracking_state(from_volume)
        for vol in range(from_volume, self.volume_num + 1):
            self.track_one_vol(vol, fig, ax)
        return None

    def _reset_tracking_state(self, from_volume):
        """
        Remove the tracking history after a specific volume to re-track from this volume
        """
        assert from_volume >= 2, "from_volume should >= 2"
        current_vol = len(self.history.r_displacements)
        del self.history.r_displacements[from_volume - 1:]
        del self.history.r_segmented_coordinates[from_volume - 1:]
        del self.history.r_tracked_coordinates[from_volume - 1:]
        del self.history.cell_phenotypes[from_volume - 1:]
        assert len(self.history.r_displacements) == from_volume - 1, \
            f"Currently data has been tracked until vol {current_vol}, the program cannot start from {from_volume}"

    def track_one_vol(self, target_volume, fig, axc6):
        """
        Track one volume of cells and update the coordinates information

        Parameters
        ----------
        target_volume : int
            The target volume to be tracked
        fig : matplotlib.figure.Figure
            A figure to draw the updated tracking results
        axc6 : list of matplotlib.axes.Axes
            A list of axes to draw each sub-figures of tracking results
        """
        # make _segment of target volume
        self.segresult.update_seg_results(*self._segment(target_volume))
        save_img3ts(range(self.z_size), self.segresult.auto_segmentation,
                    self.paths.segmentation_results + "auto_segment_t%04i_z%04i.tif", target_volume, self.use_8_bit)

        # FFN + PR-GLS predictions
        list_predictions = []
        r_coor_predicted, _ = self._predict_pos_once(source_volume=target_volume-1, draw=False)
        list_predictions.append(r_coor_predicted)
        r_coor_predicted_mean = trim_mean(list_predictions, 0.1, axis=0)

        # remove cells moved to the boundaries of the 3D image
        cells_bd = self._get_cells_on_boundary(r_coor_predicted_mean)
        self.cells_on_boundary[cells_bd] = 1

        # accurate correction to get more accurate positions
        r_disp_from_vol1_updated, i_disp_from_vol1_updated = \
            self._accurate_correction(self.cells_on_boundary, r_coor_predicted_mean)

        # transform positions into images
        self.tracked_labels, phenotypes_current = self._transform_motion_to_image(self.cells_on_boundary,
                                                                                  i_disp_from_vol1_updated)

        # save tracked labels
        save_img3ts(range(0, self.z_size), self.tracked_labels,
                    self.paths.track_results + "track_results_t%04i_z%04i.tif", target_volume, self.use_8_bit)

        self._draw_matching_6panel(target_volume, axc6, r_coor_predicted_mean, i_disp_from_vol1_updated)
        fig.canvas.draw()
        plt.savefig(self.paths.anim + "track_anim_t%04i.png" % target_volume, bbox_inches='tight')

        # update and save points locations
        self.history.r_displacements.append(r_disp_from_vol1_updated)
        self.history.r_segmented_coordinates.append(self.segresult.r_center_coordinates)
        self.history.r_tracked_coordinates.append(self.r_coordinates_tracked_t0 + r_disp_from_vol1_updated)
        self.history.cell_phenotypes.append(phenotypes_current)

        return None

    def replay_track_animation(self, from_volume=2):
        """
        Replay the tracking animation based on the stored tracking process

        Parameters
        ----------
        from_volume : int
            The start volume to show the tracking process. Should be >= 2. Default: 2

        Returns
        -------
        track_anim : matplotlib.animation.ArtistAnimation
            The animation object to be showed
        """
        fig, ax = plt.subplots(figsize=(14, int(21 * self.x_size / self.y_size)), tight_layout=True)
        plt.close(fig)
        ax.axis('off')
        track_process_images = []
        for volume in range(from_volume, self.volume_num + 1):
            try:
                im = mgimg.imread(self.paths.anim + "track_anim_t%04i.png" % volume)
            except FileNotFoundError:
                continue
            implot = ax.imshow(im)
            track_process_images.append([implot])

        track_anim = animation.ArtistAnimation(fig, track_process_images, interval=200, repeat=False).to_jshtml()
        return track_anim

    def save_coordinates(self):
        """Save 3D coordinates in a csv file under the track_information folder
        Notes
        -----
        x,y are coordinates with pixel unit, while z is the interpolated coordinate with the same unit as x and y
        """
        coord = np.asarray(self.history.r_tracked_coordinates)
        phenotypes = np.asarray(self.history.cell_phenotypes)
        t, cell, pos = coord.shape
        coord_table = np.column_stack(
            (np.repeat(np.arange(1, t + 1), cell), np.tile(np.arange(1, cell + 1), t), coord.reshape(t * cell, pos),
             phenotypes.reshape(t * cell, 1)))
        coord_table[:,4] /= self.z_xy_ratio
        np.savetxt(os.path.join(self.paths.tracking_information, "tracked_coordinates.csv"), coord_table, delimiter=',',
                   header="t,cell,x,y,z,phenotypes", comments="")
        print("Cell coordinates were stored in ./track_information/tracked_coordinates.csv")