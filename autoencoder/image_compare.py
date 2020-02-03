"""
Comparer
Compare two of the images.

Licensed under the GPL-3.0 License (see LICENSE for details)
Written by NPU-Franklin@Github.com
"""
import os
import cv2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from keras import backend as K

from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, UpSampling2D
from keras.callbacks import TensorBoard

# Requires TensorFlow 2.0.0 + and Keras 2.2.4+.
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) <= LooseVersion("1.3.0"):
    print("WARNING: Your tensorflow version might be a little bit low to run it!")
if LooseVersion(keras.__version__) <= LooseVersion("2.0.8"):
    print("WARNING: Your keras version might be a little bit low to run it!")

# Set font size and figure size.
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = (18, 24)

# Set CUDA Visual Device.
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

# Set numpy print threshold.
np.set_printoptions(threshold=1000000)

# Set size of the image.
img_width, img_height, channels = 1536, 2048, 3
input_shape = (img_height, img_width, channels)

# Define a function to resize the images.
def resize_data(dir):
    files = ["%s/%s" % (dir, num) for num in os.listdir(dir)]
    for file in files:
        img_old = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img_new = cv2.resize(img_old, (1536, 2048), interpolation=cv2.INTER_CUBIC)
        save_path = dir+'_resize/'
        img_name = os.path.basename(file)
        if os.path.exists(save_path):
            print(file)
            try:
                save_img = save_path + img_name
                cv2.imwrite(save_img, img_new, [int(cv2.IMWRITE_PNG_COMPRESSION),1])
                print('succeed!')
            except:
                print('False!')
        else:
            os.mkdir(save_path)
            print(file)
            try:
                save_img = save_path + img_name
                cv2.imwrite(save_img, img_new, [int(cv2.IMWRITE_PNG_COMPRESSION),1])
                print('succeed!')
            except:
                print('False!')

# Define a function to load data.
def load_data(dir):
    if not os.path.exists(dir):
        dir = "." + dir
    resize_data(dir)
    dir = dir + '_resize'
    files = ["%s/%s" % (dir, num) for num in os.listdir(dir)]
    arr = np.empty((len(files), img_height, img_width, channels), dtype=np.float64)
    for i, imgfile in enumerate(files):
        img = load_img(imgfile)
        x = img_to_array(img).reshape(img_height, img_width, channels)
        x = x.astype('float64')/255
        arr[i] = x
    return arr

# Load data.
test_data = load_data(r'./datasets/304x304')
print(test_data.shape)

# Load trained model.
if os.path.exists('./results/model/CAE_304x304_reshaped.h5'):
    model = load_model('./results/model/CAE_304x304_reshaped.h5')
else:
    model = load_model('../results/model/CAE_304x304_reshaped.h5')

# Get encoder part and its outputs.
encoder = K.function([model.layers[0].input],model.layers[7].output)
encoder_output = encoder([test_data])
encoder_output = encoder_output.reshape(len(encoder_output), 49152)

# Calculate R_entropy
def R_entropy(f, s):
    KL = 0
    for i, j in zip(encoder_output[f,:], encoder_output[s,:]):
        if (1 / 2 * i + 1 / 2 * j) != 0:
            if i != 0:
                KL += i * np.log(i / (1 / 2 * i + 1 / 2 * j))
    print("\n-----------------------------------------------------------------------")
    print("The smaller the characteristic value is the similar two images are.")
    print("(characteristic value >= 0)")
    print("-----------------------------------------------------------------------")
    print("\ncharacteristic value: %.10f"%KL)
    print("\n")


print("\nPlease input the first image you want to compare: ")
f = int(input())
print("Please input the second image you want to compare:")
s = int(input())

R_entropy(f, s)