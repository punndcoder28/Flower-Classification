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

# with tf.device('/device:XLA_GPU:0'):
train_directory = '/home/puneeth/Desktop/projects/Flower-Classification/data/training/'
validation_directory = '/home/puneeth/Desktop/projects/Flower-Classification/data/validation/'
test_directory = '/home/puneeth/Desktop/projects/Flower-Classification/data/test/'
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
        vertical_flip=True)
    
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_directory, target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_directory(validation_directory, target_size=TARGET_SIZE,
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

history = model.fit_generator(
        generator = train_generator,
        steps_per_epoch=13047/32,
        epochs=100,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[EarlyStopping(min_delta=0.01)]
)

model.save('vgg16.h5')

submission = pd.read_csv('/home/puneeth/Desktop/projects/Flower-Classification/data/sample_submission.csv')

test_id = []
test_pred = []

for i in submission.image_id:
    img =  cv2.resize(cv2.imread('/home/puneeth/Desktop/projects/Flower-Classification/data/test/'+str(i)+'.jpg'), (100, 100))
    img = np.expand_dims(img, axis=0)
    test_id.append[i]
    test_pred.append(int(model.predict_classes(img)))

final_submission = pd.DataFrame({'image_id': test_id, 'category': test_pred})
final_submission.head()

final_submission.to_csv('final_submission.csv', index=False)