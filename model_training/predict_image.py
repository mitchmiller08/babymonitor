"""
This module will use a trained model to predict an image's classification

Module author: Mitch Miller <mitch.miller08@gmail.com>

"""
import pickle
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

## Set model path
MODEL_PATH = 'trained_model_2018-08-31.h5'
IMG_PATH = 'data/validation/idle/test_frame_68_s&p.jpg'

model = load_model(MODEL_PATH)

img = load_img(IMG_PATH)
x = img_to_array(img)
x_rescale = x * 1./255
x_rescale = np.expand_dims(x_rescale, axis=0)

prediction = model.predict_classes(x_rescale)
probability = model.predict_proba(x_rescale)

with open('label_map_2018-08-31.pkl','rb') as label_file:
    label_map = pickle.load(label_file)

prediction_label = label_map[prediction[0]]

print("Predicted class = {} with {:.4f} probability".format(prediction_label, max(probability[0])))
