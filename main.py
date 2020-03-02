import sys
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os

with tf.device('/device:XLA_GPU:0'):
    train_directory = '/home/puneeth/Desktop/projects/Flower-Classification/data/train/*'
    test_directory = '/home/puneeth/Desktop/projects/Flower-Classification/data/test/*'
    BATCH_SIZE = 32
    TARGET_SIZE = (100, 100)

    train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
        vertical_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_directory, target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE)

    test_generator = test_datagen.flow_from_directory(test_directory, target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE)

    vgg = VGG16(weights='imagenet', include_top=True)

    vgg.summary()

    model = Sequential()
    for layer in vgg.layers[:-1]:
        model.add(layer)

    for layer in model.layers[:]:
        layer.trainable = False

    model.add(Dense(102, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()