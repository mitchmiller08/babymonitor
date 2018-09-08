"""This module generates a camera object for live streaming.

A model trained with model_training.train_model.py is loaded along with a label
mapping dictionary. The hardware camera is connected with OpenCV and inital
frame is captured to establish state.

# TODO: Rescale frames to 150x150 for prediction
        Figure out sharing frame w/ prediction process

"""
import cv2
import pickle
import numpy as np

from multiprocessing import Process, Value, Array

from keras.models import load_model

class VideoCamera(object):
    """
    Placeholder docstring

    """
    PREDICTOR_WAIT = 1

    def __init__(self, model_path, label_map_path):
        self.model = load_model(model_path)
        with open(label_map_path, 'rb') as label_file:
            self.label_map = pickle.load(label_file)

        self.current_frame = Array('d')
        self.video = cv2.VideoCapture(0)
        _, self.current_frame = self.video.read()

        self.state = Value('i')
        self.predictor = Process(target=self.predict_state)
        self.predictor.start()

    def __del__(self):
        self.predictor.terminate()
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def predict_state(self):
        while True:
            X = self.current_frame
            X_rescale = x * 1./255
            X_rescale = np.expand_dims(X_rescale, axis=0)
            self.state.value = self.model.predict_classes(X_rescale)
            time.sleep(self.PREDICTOR_WAIT)
