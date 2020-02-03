# UI_Extraction for Image Convolutional Autoencode and Compare [^Author]
This is an implementation of  Autoencoder and image comparison on Python 3, Keras, and Tensorflow. It's based on traditional Autoencoder and Relative entropy. Licensed under GPL-3.0 License (see LICENSE for details)

The repository includes:

* Source and Training code of Convolutional Autoencoder based on traditional Autoencoder.
* Pre-trained weights for CAE.
* Jupyter notebook to specify every step.

##  Getting Started

* [Extract.ipynb](notebooks/Extract.ipynb) Is the easiest way to get started. It shows detailed preprocessing and training steps.
* [run.py](autoencodr/run.py) includes source and training code of Convolutional Autoencoder based on [dataset](datasets/304x304).
* [image_compare.py](autoencoder/image_compare.py) includes code to compare two images.
* [datasets](datasets/) includes two sets of data I provide.
* [results/model](results/model) includes two pre-trained weights. 

## Requirements

* os
* numpy
* matplotlib
* tensorflow>=1.3.0
* keras>=2.0.8
* opencv-python
* IPython[all]

[^Author]:NPU-Franklin@Github.com

***Time: 2020-2-3 16:25:35*** 