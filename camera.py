"""This module generates a camera object for live streaming.

A model trained with model_training.train_model.py is loaded along with a label
mapping dictionary. The hardware camera is connected with OpenCV and inital
frame is captured to establish state. State is recalculated after a fixed
number of frames have been collected. A jpg encoded image is returned from every
call of VideoCamera.get_frame().

# TODO: Rescale frames to 150x150 for prediction

Module author: Mitch Miller <mitch.miller08@gmail.com>

"""
import cv2
import pickle
import numpy as np

from keras.models import load_model

class VideoCamera(object):
    """A class for collecting images from a hardware camera

    Attributes:
        model (:obj:keras.Sequential): The classification model used to predict
            image state.
        label_map (dict): A dictionary describing the mapping from predicted
            values to their corresponding label.

        capture (:obj:cv2.VideoCapture): The OpenCV video capture object.
        current_frame (:obj:numpy.ndarray): A numpy array of image pixel values.
            Three channels, one per RGB.

        state (int): An integer representing the state predicted by the
            classifier.
        state_label (str): A text label of the current state.
        STATE_REFRESH (int): How many frames to pass before recalculating state.

    """
    STATE_REFRESH = 60

    def __init__(self, model_path, label_map_path):
        self.model = load_model(model_path)
        with open(label_map_path, 'rb') as label_file:
            self.label_map = pickle.load(label_file)

        self.capture = cv2.VideoCapture(0)
        self._frame_count = 0
        _, self.current_frame = self.capture.read()
        self._predict_state()

    def __del__(self):
        self.capture.release()

    def get_frame(self):
        """Get the current image from the hardware camera.

        Read the capture object for the current image. If the necessary number
        of frames have already been collected, recalculate the current state.

        """
        _, self.current_frame = self.capture.read()
        self._frame_count += 1
        if self._frame_count == self.STATE_REFRESH:
            self._predict_state()
            self._frame_count = 0
        _, jpeg = cv2.imencode('.jpg', self.current_frame)
        return jpeg.tobytes()

    def _predict_state(self):
        """Predict the current state using the classification model."""
        X = self.current_frame
        X_rescale = x * 1./255
        X_rescale = np.expand_dims(X_rescale, axis=0)
        self.state = self.model.predict_classes(X_rescale)
        self.state_label = self.label_map[self.state]
