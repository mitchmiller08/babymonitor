"""This module is used to generate labeled training images from video files.

A video file is loaded and split into individual frames. Each frame will have a
set of distortions applied to it and the resulting images will be saved back to
disk for model training.

data/
    train/
        idle/
            idle001.jpg
            idle002.jpg
            ...
        normal/
            normal001.jpg
            normal002.jpg
            ...
        warn/
            warn001.jpg
            warn002.jpg
            ...
    validation/
        idle/
            idle001.jpg
            idle002.jpg
            ...
        normal/
            normal001.jpg
            normal002.jpg
            ...
        warn/
            warn001.jpg
            warn002.jpg
            ...
    source/
        idle/
            idle001.mp4
            idle002.mp4
            ...
        normal/
            normal001.mp4
            normal002.mp4
            ...
        warn/
            warn001.mp4
            warn002.mp4
            ...

Module author: Mitch Miller <mitch.miller08@gmail.com>

"""
import cv2
import os
import numpy as np

class VideoFile(object):
    """Placeholder docstring

    """
    def __init__(self, path):
        self.train_test_split = 0.75
        self.path = path
        self.dir, self.name = os.path.split(path)
        self.capture = cv2.VideoCapture(path)

    def split_frames(self):
        """Split a video into individual frames and apply noise.

        Apply gaussian additive and "salt & pepper" noise to each frame. Save
        each set of three images back to disk.

        """
        success, image = self.capture.read()
        count = 0
        while success:
            gaussian_image, salt_pepper_image = self._distort_image(image)
            normal_dir = self._train_validation_split(self.dir)
            gaussian_dir = self._train_validation_split(self.dir)
            salt_pepper_dir = self._train_validation_split(self.dir)
            # print("Saving {}/{}_frame_{}_normal.jpg".format(normal_dir,self.name[:-4],count))
            cv2.imwrite("{}/{}_frame_{}_normal.jpg"\
                        .format(normal_dir,self.name[:-4],count),image)
            cv2.imwrite("{}/{}_frame_{}_gauss.jpg"\
                        .format(gaussian_dir,self.name[:-4],count),
                        gaussian_image)
            cv2.imwrite("{}/{}_frame_{}_s&p.jpg"\
                        .format(salt_pepper_dir,self.name[:-4],count),
                        salt_pepper_image)
            success, image = self.capture.read()
            count += 1

    @staticmethod
    def _distort_image(image):
        """Apply gaussian and salt & pepper noise to an image.

        Arguments:
            image (opencv image): The image to be distorted.

        Returns:
            gaussian_image (opencv image): The image with gaussian-distributed
                additive noise.
            salt_pepper_image (opencv image): The image with random pixels
                replaced with 0 or 1.

        """
        rows, columns, channels = image.shape

        ## Add gaussian noise
        MEAN = 0
        VARIANCE = 0.001
        STANDARD_DEVIATION = np.sqrt(VARIANCE)

        gauss = np.random.normal(MEAN, STANDARD_DEVIATION,
                                (rows, columns, channels))
        gauss = gauss.reshape(rows, columns, channels)
        gaussian_image = image + (gauss * 255)

        ## Add salt & pepper noise
        SALT_VS_PEPPER = 0.5
        DENSITY = 0.005
        salt_pepper_image = np.copy(image)
        ## Salt
        n_salt = np.ceil(DENSITY * image.size * SALT_VS_PEPPER)
        coordinates = [np.random.randint(0, i-1, int(n_salt))
                       for i in image.shape[:2]]
        ## Create x-y pair for all three color channels
        coordinates[0] = np.concatenate((coordinates[0],coordinates[0],
                                         coordinates[0]))
        coordinates[1] = np.concatenate((coordinates[1],coordinates[1],
                                         coordinates[1]))
        ## Make channel array: array(0,..n..0,1,..n..,1,2,..n..,2)
        channel_array = np.concatenate([np.full(int(n_salt),i)
                                        for i in range(image.shape[2])])
        coordinates.append(channel_array)
        salt_pepper_image[tuple(coordinates)] = 255

        ## Pepper
        n_pepper = np.ceil(DENSITY * image.size * (1. - SALT_VS_PEPPER))
        coordinates = [np.random.randint(0, i-1, int(n_pepper))
                       for i in image.shape[:2]]
        ## Create x-y pair for all three color channels
        coordinates[0] = np.concatenate((coordinates[0],coordinates[0],
                                         coordinates[0]))
        coordinates[1] = np.concatenate((coordinates[1],coordinates[1],
                                         coordinates[1]))
        ## Make channel array: array(0,..n..0,1,..n..,1,2,..n..,2)
        channel_array = np.concatenate([np.full(int(n_pepper),i)
                                        for i in range(image.shape[2])])
        coordinates.append(channel_array)
        salt_pepper_image[tuple(coordinates)] = 0

        return gaussian_image, salt_pepper_image

    def _train_validation_split(self, dir):
        """Assign a file to either train or validation directory

        Arguments:
            dir (str): The base directory of the video file

        Returns:
            (str): The new directory for an image assigned to the train or
                validation set.

        """
        if np.random.random() < self.train_test_split:
            split_path = dir.split(os.sep)
            split_path[-2] = 'train'
            return os.sep.join(split_path)
        else:
            split_path = dir.split(os.sep)
            split_path[-2] = 'validation'
            return os.sep.join(split_path)
