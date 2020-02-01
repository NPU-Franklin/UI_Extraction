import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

from scipy import *
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, UpSampling2D
from keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import plot_model

plt.rcParams['font.size'] = 8
plt.rcParams['figure.figsize'] = (18,24)

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

img_width, img_height, channels = 1536, 2048, 3
input_shape = (img_height, img_width, channels)

def load_data():
    dir = './datasets'
    files = ["%s/%s" % (dir, num) for num in os.listdir(dir)]
    arr = np.empty((len(files), img_height, img_width, channels), dtype=np.float64)
    for i, imgfile in enumerate(files):
        img = load_img(imgfile)
        x = img_to_array(img).reshape(img_height, img_width, channels)
        x = x.astype('float64')/255
        arr[i] = x
    return arr

train_data = load_data()
print(train_data.shape)

np.save('./data_numpyarray/dataset.npy', train_data)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_data, train_data, epochs=100, shuffle=True, batch_size=1)
model.save('./results/model/CAE.h5')
