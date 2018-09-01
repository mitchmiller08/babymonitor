"""
This module will train an image classifier model.

The general architecture is built based on this
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
project by Francois Chollet.

We train a small convnet on image files generated
from indvidiual frames of video files. The images are duplicated and have
random noise applied to mimic poor video performance during live capture. This
step has an added benefit of a regularization effect [Ref 2]. We then apply a
second round of distortions to augment the number of training examples.

Module author: Mitch Miller <mitch.miller08@gmail.com>

References:
    [1] Chollet, Francois. Building powerful image classification models using
        very little data. The Keras Blog, 2016
    [2] Goodfellow, Ian, Shlens, Jonathon, and Szegedy, Christian. Explaining
        and Harnessing Adversarial Examples. ICLR, 2015

"""
import os

from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

## Define image dimensions
img_width, img_height = 150, 150

## Define training parameters
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
m_train = sum(len(files) for root, dir, files in os.walk(train_data_dir))
m_validation = sum(len(files) for root, dir, files in
                   os.walk(validation_data_dir))
epochs = 50
batch_size = 16

## Ensure image is formatted correctly
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

## Define model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

## Compile model
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

## Define data distortion generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size
    class_mode='categorical')

validation_datagen = ImageDataGenerator(
    rescale=1./255) # Only need to rescale

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size
    class_mode='categorical')

## Train model
model.fit_generator(
    train_generator,
    steps_per_epoch=m_train//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=m_validation//batch_size)

## Save model
current_date = datetime.now().date()
model.save_weights('trained_model_{}.h5'.format(current_date))
