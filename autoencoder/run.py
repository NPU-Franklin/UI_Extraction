"""
Autoencoder
Model compile and train.

Licensed under the GPL-3.0 License (see LICENSE for details)
Written by NPU-Franklin@Github.com
"""
import os
import cv2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, UpSampling2D
from keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import plot_model

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
train_data = load_data(r'./datasets/304x304')
print(train_data.shape)

# Save data array as numpy array.
if os.path.exists(r"./data_numpyarray"):
    np.save('./data_numpyarray/dataset_304xx304_reshaped.npy', train_data)
else:
    np.save('../data_numpyarray/dataset_304xx304_reshaped.npy', train_data)

# Define model
model = Sequential()

# encoder: （Conv2D + AveragePooling2D)*4
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

# decoder: (UpSampling2D + Conv2D)*4 + conv2D
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train
model.fit(train_data, train_data, epochs=100, shuffle=True)

# Save model
if os.path.exists(r"./results"):
    model.save('./results/model/CAE_304x304_reshaped.h5')
else:
    model.save('../results/model/CAE_304x304_reshaped.h5')
