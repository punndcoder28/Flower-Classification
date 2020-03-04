from collections import Counter
from itertools import chain
from skimage import io
import glob, cv2, os
import numpy as np

train_dir = '/home/puneeth/Desktop/projects/Flower-Classification/data/train/*.jpg'
test_dir = '/home/puneeth/Desktop/projects/Flower-Classification/data/test/*.jpg'

def ret_size(image):
    img = cv2.imread(image, 0) # load image in greyscale mode
    return img.shape

def check_size():
    '''
        To check the shape of images and to see if any reshaping is needed
    '''
    print("Checking size of images")
    size = []

    for filename in glob.glob(dir):
        size.append(ret_size(filename))

    o = Counter(chain(*size))
    print(o)
    '''
        All images are 500x500 and RGB
    '''

def train_resize():
    '''
        Resize images to 100x100 and save it in the form of a greyscale image

        Returns: Resized images
    '''
    print("Resizing the train images")
    scale_percent = 0.448

    for filename in glob.glob(train_dir):
        original_img = cv2.imread(filename)
        width = int(original_img.shape[0]*scale_percent)
        height = int(original_img.shape[1]*scale_percent)
        dim = (width, height)
        smaller = cv2.resize(original_img, dim, interpolation=cv2.INTER_AREA)
        os.remove(filename)
        cv2.imwrite(filename, smaller)

def test_resize():
    '''
        Resize images to 100x100 and save it in the form of a greyscale image

        Returns: Resized images
    '''
    print("Resizing the test images")
    scale_percent = 0.

    for filename in glob.glob(test_dir):
        original_img = cv2.imread(filename, 0)
        width = int(original_img.shape[0]*scale_percent)
        height = int(original_img.shape[1]*scale_percent)
        dim = (width, height)
        smaller = cv2.resize(original_img, dim, interpolation=cv2.INTER_AREA)
        os.remove(filename)
        cv2.imwrite(filename, smaller)



if __name__=='__main__':
    # check size of all the images
    # check_size()

    # resize images to 100X100 images in greyscale
    # resize()

    # convert the images to an array for training
    # x_train = convert_images_to_array()
    # print(x_train.shape)

    # convert labels to array
    # y_train = convert_labels_to_array()
    # test_resize()