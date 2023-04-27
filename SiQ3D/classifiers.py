"""
A module for training the classifiers used for cell phenotyping.
Author: Simon lbd
"""

import os
import random
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tifffile import imread
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Dropout, Flatten, Dense, \
    concatenate, LeakyReLU
from tensorflow.keras import Input, Model, optimizers, regularizers
from .preprocess import _make_folder, _get_files
TITLE_STYLE = {'fontsize': 16, 'verticalalignment': 'bottom'}


def load_image(image_folder):
    """
    Load images stored in the corresponding folder.

    Parameters
    ----------
        image_folder: str, the folder storing the images

    Return
    ----------
        image: numpy.ndarray, shape (width, height, depth)
    """
    image_path = _get_files(image_folder)
    image = []
    for img_path in image_path:
        image.append(imread(img_path).transpose((1, 2, 0)))
    image = np.array(image)
    print("Load data set with shape:", image.shape)
    return image


def rotate(volume):
    """
    Rotate the volume by random degrees.

    Parameters
    ----------
        volume: numpy.ndarray, the training data
    """

    def scipy_rotate(volume):
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 255] = 255
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.uint8)
    return augmented_volume


def train_preprocessing(volume, label):
    """
    Process training data by rotating and adding a channel.

    Parameters
    ----------
        volume: numpy.ndarray, the images
        label: numpy.ndarray, the corresponding labels

    Return
    ----------
        volume: numpy.ndarray, the augmented images
        label: numpy.ndarray, the corresponding labels
    """
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocess(volume, label):
    """
    Process validation data by only adding a channel.

    Parameters
    ----------
        volume: numpy.ndarray, the images
        label: numpy.ndarray, the corresponding labels

    Return
    ----------
        volume, label: numpy.ndarray
    """
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def multi_scale_classifier(width=64, height=64, depth=27):
    """
    A multiscale classifier for identifying cell phenotypes.

    Parameters
    ----------
        width: int, the width of the input image
        height: int, the height of the input image
        depth: int, the depth of the input image

    Return
    ----------
        model: keras.model
    """
    inputs = Input((width, height, depth, 1))
    conv1 = Conv3D(filters=8, kernel_size=(3, 3, 1), padding="same", activation="relu")(inputs)
    conv1 = Conv3D(filters=8, kernel_size=(3, 3, 1), padding="same", activation="relu")(conv1)
    conv1 = MaxPool3D()(conv1)

    conv2_1 = Conv3D(filters=8, kernel_size=(3, 3, 1), padding="same")(conv1)
    conv2_2 = Conv3D(filters=8, kernel_size=(3, 3, 2), padding="same")(conv1)
    conv2_3 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding="same")(conv1)

    conv2 = concatenate([conv2_1, conv2_2, conv2_3], axis=3)
    conv2 = LeakyReLU()(conv2)

    conv3_1 = Conv3D(filters=16, kernel_size=(3, 3, 1), padding="same")(conv2)
    conv3_2 = Conv3D(filters=16, kernel_size=(3, 3, 2), padding="same")(conv2)
    conv3_3 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding="same")(conv2)

    conv3 = concatenate([conv3_1, conv3_2, conv3_3], axis=3)
    conv3 = LeakyReLU()(conv3)

    conv4 = Conv3D(filters=16, kernel_size=(3, 3, 1), padding="same", activation="relu")(conv3)
    conv4 = Conv3D(filters=16, kernel_size=(3, 3, 1), padding="same", activation="relu")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool1 = MaxPool3D()(conv4)
    drop1 = Dropout(0.2)(pool1)
    dens1 = Flatten()(drop1)
    dens2 = Dense(units=256, activation="relu")(dens1)
    dens2 = Dense(units=96, activation="relu")(dens2)
    outputs = Dense(units=1, activation="sigmoid")(dens2)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def sequential_classifier(width=64, height=64, depth=27):
    """
        A classic 3D convolutional classifier for identifying cell phenotypes.

        Parameters
        ----------
            width: int, the width of the input image
            height: int, the height of the input image
            depth: int, the depth of the input image

        Return
        ----------
            model: keras.model
        """
    inputs = Input((width, height, depth, 1))
    conv1 = Conv3D(filters=8, kernel_size=(3, 3, 1), padding="same", activation="relu")(inputs)
    conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(filters=8, kernel_size=(3, 3, 1), padding="same", activation="relu")(conv1)
    #conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool3D(pool_size=2)(conv1)

    conv2 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding="same", activation="relu")(pool1)
    conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(filters=16, kernel_size=3, padding="same", activation="relu")(conv2)
    #conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool3D()(conv2)

    drop1 = Dropout(0.2)(pool2)
    dens1 = Flatten()(drop1)

    dens2 = Dense(units=96, activation="relu")(dens1)
    outputs = Dense(units=1, activation="sigmoid")(dens2)

    model = Model(inputs=inputs, outputs=outputs)
    return model


class Classifier:
    """
    A class for training classifier.

    Attributes
    ----------
    """
    def __init__(self, model, folder_path, learning_rate=0.001):
        self.model = model
        self.folder_path = folder_path
        self.data_set0 = None
        self.data_set1 = None
        self.label0 = None
        self.label1 = None
        self.train_image = None
        self.validate_image = None
        self.train_label = None
        self.validate_label = None
        self.image_path0 = None
        self.image_path1 = None
        self.train_loader = None
        self.validation_loader = None
        self.val_acc = None
        self.acc = None
        self.loss = None
        self.val_loss = None
        self.best_epoch = None
        self.models_path = ""
        self._make_folders()
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=learning_rate), metrics=["acc"])
        self.model.save_weights(os.path.join(self.models_path, 'weights_initial.h5'))

    def _make_folders(self):
        """
        Make folders for storing data and results
        """
        print("Made folders under:", os.getcwd())
        folder_path = self.folder_path
        self.image_path1 = _make_folder(os.path.join(folder_path, "image1/"))
        self.image_path0 = _make_folder(os.path.join(folder_path, "image0/"))
        self.models_path = _make_folder(os.path.join(folder_path, "models/"))

    def load_dataset(self):
        """
        Load dataset from the corresponding folders, make corresponding labels
        """
        self.data_set0 = load_image(self.image_path0)
        self.data_set1 = load_image(self.image_path1)
        self.label0 = np.array([0 for _ in range(self.data_set0.shape[0])])
        self.label1 = np.array([1 for _ in range(self.data_set1.shape[0])])

    def separate_train_valid_data(self, val_ratio=0.2):
        """
        Separate the train data and validation data
        """
        train_num0 = int(self.data_set0.shape[0] * (1 - val_ratio))
        train_num1 = int(self.data_set1.shape[0] * (1 - val_ratio))
        self.train_image = np.concatenate((self.data_set0[:train_num0], self.data_set1[:train_num1]),
                                          axis=0)
        self.validate_image = np.concatenate((self.data_set0[train_num0:], self.data_set1[train_num1:]),
                                             axis=0)
        self.train_label = np.concatenate((self.label0[:train_num0], self.label1[:train_num1]),
                                          axis=0)
        self.validate_label = np.concatenate((self.label0[train_num0:], self.label1[train_num1:]),
                                             axis=0)

    def _augment_data(self, batch_size=2):
        """
        Apply data augmentation

        Parameters
        ----------
            batch_size: int, batch size
        """
        self.train_loader = tf.data.Dataset.from_tensor_slices((self.train_image, self.train_label))
        self.validation_loader = tf.data.Dataset.from_tensor_slices((self.validate_image, self.validate_label))
        self.train_loader = (self.train_loader.shuffle(self.train_image.shape[0]).map(
            train_preprocessing).batch(batch_size).prefetch(2))
        self.validation_loader = (self.validation_loader.shuffle(self.validate_image.shape[0]).map(
            validation_preprocess).batch(batch_size).prefetch(2))

    def preprocess(self, val_ratio=0.2, batch_size=2):
        """
        Separate train data and validation data, apply data augmentation.

        Parameters
        ----------
            val_ratio: float, the ratio of validation data
            batch_size: int, batch size
        """
        self.separate_train_valid_data(val_ratio)
        self._augment_data(batch_size=batch_size)

    def train(self, iteration=100, weights_name="weights_training_"):
        """
        Train the classifier

        Parameters
        ----------
            iteration: int, the number of epochs to train the model. Default: 100
            weights_name: str, the prefix of the weights files to be stored during training.
        """
        self.model.load_weights(os.path.join(self.models_path, 'weights_initial.h5'))
        for step in range(1, iteration + 1):
            self.model.fit(self.train_loader, validation_data=self.validation_loader,
                           epochs=1, verbose=2)
            if step == 1:
                self.val_acc = [self.model.history.history["val_acc"][-1]]
                self.acc = [self.model.history.history["acc"][-1]]
                self.val_loss = [self.model.history.history["val_loss"][-1]]
                self.loss = [self.model.history.history["loss"][-1]]
                print("val_acc at step 1: ", max(self.val_acc))
                self.model.save_weights(os.path.join(self.models_path, weights_name + f"step{step}.h5"))
                self.best_epoch = step
            else:
                val_acc = self.model.history.history["val_acc"][-1]
                if val_acc > max(self.val_acc):
                    print("At step: ", step, ",val_acc updated from ", max(self.val_acc), " to ", val_acc)
                    self.model.save_weights(os.path.join(self.models_path, weights_name + f"step{step}.h5"))
                    self.best_epoch = step

                self.val_acc.append(self.model.history.history["val_acc"][-1])
                self.acc.append(self.model.history.history["acc"][-1])
                self.loss.append(self.model.history.history["loss"][-1])
                self.val_loss.append(self.model.history.history["val_loss"][-1])

    def select_weights(self, epoch, weights_name="weights_training_"):
        """
        Select the weights that have the best performance in cell phenotypes classification

        Parameters
        ----------
            epoch: int, the step corresponding to the best classification
            weights_name: str, the prefix of the weights file to be restored.
        """
        self.model.load_weights(os.path.join(self.models_path, weights_name + f"step{epoch}.h5"))
        self.model.save(os.path.join(self.models_path, "classifier_pretrained.h5"))

    def draw_loss_accuracy(self):
        """
        Draw the loss/val_loss and acc/val_acc
        """
        fig, ax = plt.subplots(1, 2, figsize=(16, 10))
        ax[0].plot(self.loss, 'b', linewidth=2, label='loss')
        ax[0].plot(self.val_loss, 'r', linewidth=2, label='val_loss')
        ax[0].set_title("Loss/val_loss", fontdict=TITLE_STYLE)
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[1].plot(self.acc, 'b', linewidth=2, label='accuracy')
        ax[1].plot(self.val_acc, 'r', linewidth=2, label="val_accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        plt.tight_layout()

    def save_loss_accuracy(self):
        """
        Save the loss and accuracy during the training procedure
        """
        loss_acc = np.column_stack((np.array(self.loss), np.array(self.val_loss), np.array(self.acc),
                                    np.array(self.val_acc)))
        np.savetxt(os.path.join(self.models_path, "losses_and_accuracies.csv"), loss_acc,
                   delimiter=',', header="loss,val_loss,acc,val_acc", comments="")