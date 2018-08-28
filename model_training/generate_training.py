"""This module is used to generate labeled training images from video files.

A video file is loaded and split into individual frames. Each frame will have a
set of distortions applied to it and the resulting images will be saved back to
disk for model training.

Module author: Mitch Miller <mitch.miller08@gmail.com>

"""
import cv2
import os
import numpy as np

class VideoFile(object):
    """Placeholder docstring

    """
    def __init__(self, path):
        self.path = path
        self.name = path.split(os.sep)[-1]
        self.capture = cv2.VideoCapture(path)

    def split_frames(self):
        """Split a video into individual frames and apply noise.

        Apply gaussian additive and "salt & pepper" noise to each frame. Save
        each set of three images back to disk.
        """
        success, image = self.capture.read()
        count = 0
        while success:
            gaussian_image, salt_pepper_image = self.distort_image(image)
            cv2.imwrite("{}_frame_{}_normal.jpg".format(self.name,count),
                        image)
            cv2.imwrite("{}_frame_{}_gauss.jpg".format(self.name,count),
                        gaussian_image)
            cv2.imwrite("{}_frame_{}_s&p.jpg".format(self.name,count),
                        salt_pepper_image)
            success, image = self.capture.read()
            count += 1

    @staticmethod
    def distort_image(image):
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
        VARIANCE = 0.1
        STANDARD_DEVIATION = np.sqrt(VARIANCE)

        gauss = np.random.normal(MEAN, STANDARD_DEVIATION,
                                (rows, columns, channels))
        gauss = gauss.reshape(rows, columns, channels)
        gaussian_image = image + gauss

        ## Add salt & pepper noise
        SALT_VS_PEPPER = 0.5
        DENSITY = 0.005
        salt_pepper_image = np.copy(image)
        ## Salt
        n_salt = np.ceil(DENSITY * image.size * SALT_VS_PEPPER)
        coordinates = [np.random.randint(0, i-1, int(n_salt))
                       for i in image.shape]
        salt_pepper_image[coordinates] = 1
        ## Pepper
        n_pepper = np.ceil(DENSITY * image.size * (1. - SALT_VS_PEPPER))
        coordinates = [np.random.randint(0, i-1, int(n_pepper))
                       for i in image.shape]
        salt_pepper_image[coordinates] = 0

        return gaussian_image, salt_pepper_image
