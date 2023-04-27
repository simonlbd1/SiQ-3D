"""
A module for tracking NK cells.
Author: Simon lbd
"""
import os
import btrack
import matplotlib.pyplot as plt
import numpy as np
from .preprocess import _make_folder, read_image_sequence, mean_background, correct_bleaching, _normalize_image, \
    crop_subregion, save_img3ts
from .unet3d import unet3_prediction
from .segmentation import otsu_threshold, simple_threshold, watershed_tascan
from tensorflow.keras.models import load_model
TITLE_STYLE = {'fontsize': 16, 'verticalalignment': 'bottom'}


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
    Paths for storing data and results used by the Tracker instance

    Parameters
    ----------
    folder_path: str,
        the path of the folder to store all data, models and results
    image_name: str,
        the file name of the raw images
    unet_model_file: str,
        the file name of the 3D U-Net model
    classifier_file: str,
        the file name of the classifier

    Attributes
    ----------
    raw_image: str,
        the path of the folder to store the raw images to be tracked
    models: str,
        the path of the folder to store 3D U-Net and classifier files
    segmentation_results: str,
        the path of the folder to store segmentation results (segmented images, before tracking)
    track_results: str,
        the path of the folder to store tracking results (label images)
    track_information: str,
        the path of the folder to store the cell coordinates
    """
    def __init__(self, folder_path, image_name, unet_model_file, classifier_file,
                 track_config_file):
        self.folder = folder_path
        self.unet_model_file = unet_model_file
        self.classifier_file = classifier_file
        self.track_config_file = track_config_file
        self.image_name = image_name
        self.raw_image = None
        self.models = None
        self.segmentation_results = None
        self.track_results = None
        self.track_information = None
        self.segmentation_name = "segment_results_%04i_z%04i.tif"

    def make_folders(self):
        """
        Make folders for storing data, models and results
        """
        print("Following folders were made under: ", os.getcwd())
        folder_path = self.folder
        self.raw_image = _make_folder(os.path.join(folder_path, "data/"))
        self.models = _make_folder(os.path.join(folder_path, "models/"))
        self.segmentation_results = _make_folder(os.path.join(folder_path, "segmentation_results/"))
        self.track_results = _make_folder(os.path.join(folder_path, "track_results/"))
        self.track_information = _make_folder(os.path.join(folder_path, "track_information/"))


class Tracker:
    """
    A class for tracking NK cells
    """
    def __init__(self, volume_num, xyz_size, z_xy_ratio, noise_level, folder_path, image_name, classifier_file,
                 track_config_file, unet_model_file=None, min_size=200, min_distance=10, touch_ratio=0.5, touch_area=50,
                 threshold_method=0, search_radius=20, tascan_=0):
        self.volume = volume_num
        self.x_siz = xyz_size[0]
        self.y_siz = xyz_size[1]
        self.z_siz = xyz_size[2]
        self.z_xy_ratio = z_xy_ratio
        self.noise_level = noise_level
        self.min_size = min_size
        self.min_distance = min_distance
        self.touch_ratio = touch_ratio
        self.touch_area = touch_area
        self.threshold_method = threshold_method
        self.search_radius = search_radius
        self.paths = Paths(folder_path, image_name, unet_model_file, classifier_file, track_config_file)
        self.paths.make_folders()
        self.segmented_coordinates = []
        self.unet = None
        self.classifier = None
        self.mean_intensity_vol1 = None
        self.back_intensity_vol1 = None
        self.tascan_ = tascan_
        self.classifier_size = None
        self.t = []
        self.x_seg = []
        self.y_seg = []
        self.z_seg = []
        self.cell_phenotypes = []
        self.cell_labels = []
        self.cell_coordinates = None
        self.tracks = None
        self.tracked_coordinates = None
        self.first_instances = None
        self.first_frame = None
        self.load_seg = 0

    def tracking_initialize(self):
        """
        Load classifier and 3D U-Net
        """
        self.classifier = load_model(os.path.join(self.paths.models, self.paths.classifier_file))
        self.classifier_size = self.classifier.input_shape
        if self.paths.unet_model_file is not None:
            self.unet = load_model(os.path.join(self.paths.models, self.paths.unet_model_file))
        img_vol1 = read_image_sequence(1, self.paths.raw_image, self.paths.image_name, (1, self.z_siz+1))
        self.mean_intensity_vol1, self.back_intensity_vol1 = mean_background(img_vol1)

    def set_segmentation(self, min_size, min_distance, touch_ratio, touch_area, noise_level):
        """
        Reset the segmentation parameters
        """
        self.min_size = min_size
        self.first_instances = min_distance
        self.touch_ratio = touch_ratio
        self.touch_area = touch_area
        self.noise_level = noise_level
        print("Finished segmentation parameters reset.")

    def _segment(self, vol):
        """
        Segment the cells (a frame)

        Parameters
        ----------
        vol: int,
            a specific volume

        Returns
        ----------
        seg_cells: numpy.ndarray,
            the cell/non-cell regions predictions (a frame)
        bleach_correct: numpy.ndarray,
            image after photobleaching correction
        """
        raw_img = read_image_sequence(vol, self.paths.raw_image, self.paths.image_name, (1, self.z_siz+1))

        # photobleaching correction
        bleach_correct = correct_bleaching(raw_img, self.mean_intensity_vol1, self.back_intensity_vol1)

        # local contrast normalization
        norm_img = _normalize_image(bleach_correct, self.noise_level)

        # predict cell/non-cell regions
        if self.paths.unet_model_file is not None:
            norm_img = np.expand_dims(norm_img, axis=(0, 4))
            seg_cells = unet3_prediction(norm_img, self.unet)
            seg_cells = seg_cells[0,:,:,:,0]
        elif self.threshold_method:
            seg_cells = otsu_threshold(norm_img, filter_size=5)
        else:
            seg_cells = simple_threshold(raw_img, self.noise_level, filter_size=5)

        return seg_cells, raw_img

    def segment_one_vol(self, vol):
        """
        Segment one volume.

        Parameters
        ----------
        vol: int,
            the specific volume

        Returns
        ----------
        cell_instances: numpy.ndarray,
            the segmented images
        """
        seg_cells, raw_image = self._segment(vol)
        cell_instances = watershed_tascan(seg_cells, self.z_siz, min_distance_2d=3, min_distance_3d=self.min_distance,
                                          samplingrate=[1, 1, self.z_xy_ratio], min_size=self.min_size,
                                          min_touching_area=self.touch_area, min_touching_ratio=self.touch_ratio,
                                          tascan_=self.tascan_)
        return cell_instances, raw_image

    def segment_vol1(self):
        """
        Segment the first one volume
        """
        self.tracking_initialize()
        self.first_instances, self.first_frame = self.segment_one_vol(vol=1)

    def draw_segresult(self):
        """
        Draw the segmentation result of the first frame
        """
        vmax = np.percentile(self.first_frame, 99.9)
        vmin = np.percentile(self.first_frame, 10)
        fig, ax = plt.subplots(1, 2, figsize=(20, int(12 * self.x_siz / self.y_siz)))
        ax[0].imshow(np.max(self.first_frame, axis=2), vmin=vmin, vmax=vmax, cmap="gray")
        ax[0].set_title("Raw image at vol 1", fontdict=TITLE_STYLE)
        ax[1].imshow(np.max(self.first_instances, axis=2), cmap=get_random_cmap(num=np.max(self.first_instances)))
        ax[1].set_title("Auto-segmentation at vol 1", fontdict=TITLE_STYLE)
        plt.tight_layout()

    def segment_all_vol(self):
        """
        Segment all volumes
        """
        for vol in range(1, self.volume+1):
            cell_instances, raw_image = self.segment_one_vol(vol)
            save_img3ts(range(self.z_siz), cell_instances,
                        self.paths.segmentation_results + self.paths.segmentation_name, vol, use_8_bit=False)
            for label in np.unique(cell_instances):
                if label > 0:
                    sub_label_img, cell_coordinates = crop_subregion(cell_instances, raw_image,
                                                                     self.classifier_size, label)
                    sub_label_img = np.expand_dims(sub_label_img, axis=(0, 4))
                    label_phenotype = self.classifier.predict(sub_label_img)
                    label_phenotype = 1 if label_phenotype > 0.5 else 0
                    self.t.append(vol)
                    self.x_seg.append(cell_coordinates[0])
                    self.y_seg.append(cell_coordinates[1])
                    self.z_seg.append(cell_coordinates[2])
                    self.cell_phenotypes.append(label_phenotype)
                    self.cell_labels.append(label)
            if vol % 10 == 0:
                print("Finished segmentation: vol ", vol)

    def load_segmentation(self):
        """
        Load the manual segmentation results
        """
        self.t = []
        self.x_seg = []
        self.y_seg = []
        self.z_seg = []
        self.cell_phenotypes = []
        self.cell_labels = []
        for vol in range(1, self.volume+1):
            raw_img = read_image_sequence(vol, self.paths.raw_image, self.paths.image_name, (1, self.z_siz + 1))
            # bleach_correct = correct_bleaching(raw_img, self.mean_intensity_vol1, self.back_intensity_vol1)
            # process_image = _normalize_image(bleach_correct, self.noise_level)

            cell_instances = read_image_sequence(vol, self.paths.segmentation_results,
                                                 self.paths.segmentation_name, (1, self.z_siz + 1))

            for label in np.unique(cell_instances):
                if label > 0:
                    sub_label_img, cell_coordinates = crop_subregion(cell_instances, raw_img,
                                                                     self.classifier_size, label)
                    sub_label_img = np.expand_dims(sub_label_img, axis=(0, 4))
                    label_phenotype = self.classifier.predict(sub_label_img)
                    self.t.append(vol)
                    self.x_seg.append(cell_coordinates[0])
                    self.y_seg.append(cell_coordinates[1])
                    self.z_seg.append(cell_coordinates[2])
                    self.cell_phenotypes.append(label_phenotype)
                    self.cell_labels.append(label)
        self.load_seg = 1
        print("Finished loading segmentation images.")

    def _transform_layer_to_real(self):
        """
        Transform the layer coordinates to the real coordinates
        """
        self.cell_coordinates = {'t': np.array(self.t), 'x': np.array(self.x_seg)/self.z_xy_ratio,
                                 'y': np.array(self.y_seg)/self.z_xy_ratio, 'z': np.array(self.z_seg),
                                 'cell_labels': np.array(self.cell_labels),
                                 'cell_phenotypes': np.array(self.cell_phenotypes)}

    def link_cells(self):
        """
        Perform Bayesian tracking algorithm
        """
        objects = btrack.dataio.objects_from_dict(self.cell_coordinates)
        with btrack.BayesianTracker() as btracker:
            btracker.configure_from_file(os.path.join(self.paths.models, self.paths.track_config_file))
            btracker.max_search_radius = self.search_radius
            btracker.tracking_updates = ['MOTION']
            btracker.append(objects)
            btracker.volume = ((0, int(self.x_siz/self.z_xy_ratio)), (0, int(self.y_siz/self.z_xy_ratio)),
                               (0, self.z_siz))
            btracker.track_interactive(step_size=50)
            btracker.optimize()
            self.tracks = btracker.tracks

        x = []
        y = []
        z = []
        t = []
        label = []
        old_label = []
        phenotype = []
        for i in range(len(self.tracks)):
            x.extend(self.tracks[i]['x'])
            y.extend(self.tracks[i]['y'])
            z.extend(self.tracks[i]['z'])
            t.extend(self.tracks[i]['t'])
            label.extend([self.tracks[i]['ID']]*len(self.tracks[i]['x']))
            old_label.extend(list(self.tracks[i]['properties']['cell_labels']))
            phenotype.extend(list(self.tracks[i]['properties']['cell_phenotypes']))

        self.tracked_coordinates = np.column_stack((np.array(label), np.array(t), np.array(x)*self.z_xy_ratio,
                                                   np.array(y)*self.z_xy_ratio, np.array(z), np.array(phenotype)))

        label = np.array(label)
        old_label = np.array(old_label)
        t = np.array(t)

        for vol in range(1, self.volume+1):
            old_label_img = read_image_sequence(vol, self.paths.segmentation_results, self.paths.segmentation_name,
                                                (1, self.z_siz + 1))
            # old_label_img = np.load(self.paths.segmentation_results + "t%04i.npy" % vol, allow_pickle=True)
            vol_label = label[t == vol]
            vol_old_label = old_label[t == vol]
            new_label_img = np.zeros_like(old_label_img, dtype=np.int16)
            for i in range(len(vol_label)):
                new_label_img[np.where(old_label_img == vol_old_label[i])] = vol_label[i]
            save_img3ts(range(self.z_siz), new_label_img, self.paths.track_results + "track_results_t%04i_z%04i.tif",
                        vol, use_8_bit=False)
            if vol % 10 == 0:
                print("Finished writing label images at vol: ", vol)

    def save_coordinates(self):
        """
        Save the coordinates of tracked cells
        """
        np.savetxt(os.path.join(self.paths.track_information, "tracked_coordinates.csv"), self.tracked_coordinates,
                   delimiter=',', header="label,t,x,y,z,phenotype", comments="")
        print("Cell coordinates were stored in ./track_information/tracked_coordinates.csv")

    def track(self):
        """
        Implement cell tracking.
        """
        if not self.load_seg:
            self.segment_all_vol()
            print("Finished image segmentation")
        self._transform_layer_to_real()
        print("Finished coordinates transform")
        self.link_cells()
