#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 08:43:38 2017

"""

import numpy as np 
from scipy.misc import imread, imresize
from scipy import ndimage
import matplotlib.pyplot as plt

## Use ITU-R 601-2 luma transform 
## L = R * 299/1000 + G * 587/1000 + B * 114/1000
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

image = imread('1id/00.jpg')
#plt.imshow(image)
image_gray = rgb2gray(image)
plt.imshow(image_gray, cmap=plt.cm.gray) 

