import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_train = []
files = glob.glob("/home/puneeth/Desktop/projects/Flower-Classification/data/train/*.jpg")
for file in files:
    image = cv2.imread(file, 0)
    x_train.append(image)

x_train_numpy = np.asarray(x_train)

np.save("/home/puneeth/Desktop/projects/Flower-Classification/x_train_numpy", x_train_numpy)

x_test = []
files = glob.glob("/home/puneeth/Desktop/projects/Flower-Classification/data/test/*.jpg")
for file in files:
    image = cv2.imread(file, 0)
    x_test.append(image)

x_test_numpy = np.asarray(x_test)

np.save("/home/puneeth/Desktop/projects/Flower-Classification/x_test_numpy", x_test_numpy)

y_label = []
y = pd.read_csv('/home/puneeth/Desktop/projects/Flower-Classification/data/train.csv')
y_label.append(y['category'])

y_label = np.asarray(y_label)
np.save("/home/puneeth/Desktop/projects/Flower-Classification/y_label_numpy", y_label)