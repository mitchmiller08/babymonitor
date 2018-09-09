"""This module generates a camera object for live streaming.

A model trained with model_training.train_model.py is loaded along with a label
mapping dictionary. The hardware camera is connected with OpenCV and inital
frame is captured to establish state. State is recalculated after a fixed
number of frames have been collected. A jpg encoded image is returned from every
call of VideoCamera.get_frame().

Module author: Mitch Miller <mitch.miller08@gmail.com>

"""
import cv2
import pickle
import numpy as np

from multiprocessing import Process, Queue

from keras.models import load_model

from model_training.train_model import img_width, img_height

class VideoCamera(object):
    """A class for collecting images from a hardware camera

    Attributes:
        model (:obj:keras.Sequential): The classification model used to predict
            image state.

        capture (:obj:cv2.VideoCapture): The OpenCV video capture object.
        _frame_count (int): The number of frames collected since the last state
            calculation.

        frame_q (:obj:multiprocessing.Queue): A queue of frames to be used for
            state calculations
        state_q (:obj:multiprocessing.Queue): A queue of recent state
            calculation results
        predictor (:obj:multiprocessing.Process): A background process for the
            state calculations.
        STATE_REFRESH (int): How many frames to pass before recalculating state.

    """
    STATE_REFRESH = 60

    def __init__(self, model_path, label_map_path):
        ## Model attributes
        self.model = load_model(model_path)

        ## Capture attributes
        self.capture = cv2.VideoCapture(0)
        self._frame_count = 0

        ## Predictor attributes
        self.frame_q = Queue()
        self.state_q = Queue()
        self.predictor = Process(target=self._predict_state)
        self.predictor.start()

    def __del__(self):
        self.predictor.terminate()
        self.capture.release()

    def get_frame(self):
        """Get the current image from the hardware camera.

        Read the capture object for the current image. If the necessary number
        of frames have already been collected, recalculate the current state.

        """
        ## Get current frame from camera
        _, current_frame = self.capture.read()
        self._frame_count += 1
        ## Check if time to recompute state and old state calculation is done
        if self._frame_count >= self.STATE_REFRESH\
        and self.frame_q.empty():
            self.frame_q.put(current_frame)
            self._frame_count = 0
        ## Encode frame as .jpg
        _, jpeg = cv2.imencode('.jpg', current_frame)
        return jpeg.tobytes()

    def _predict_state(self):
        """Predict the current state using the classification model."""
        ## Get frame from queue
        X = self.frame_q.get()
        ## Rescale image to model parameters
        X = cv2.resize(X, dsize=(img_width, img_height),
                       interpolation=cv2.INTER_LINEAR)
        X_rescale = X * 1./255
        X_rescale = np.expand_dims(X_rescale, axis=0)
        current_state = self.model.predict_classes(X_rescale)
        ## Discard oldest state
        _ = self.state_q.get()
        self.state_q.put(current_state)
