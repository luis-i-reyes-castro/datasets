#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 08:43:38 2017

"""
import os, sys
import numpy as np 
from scipy.misc import imread, imresize
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## list_name_image is path and id_row
## 010(Name file) (0--> id row) (10--> id column)
def readfile(list_name_image, id_row):
    images = []
    for file in list_name_image:
        if file.startswith(id_row):
            image = imread(file)
            images.append(image) #<class 'numpy.ndarray'> #shape 480 640 3
    return images
            
## Use ITU-R 601-2 luma transform 
## L = R * 299/1000 + G * 587/1000 + B * 114/1000
def rgb2gray(rgb):
    image_gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return image_gray

#image_file = readfile(os.listdir(os.getcwd()), "0")

#print(image_file)
image = imread('/home/luiireye/Desktop/original-dataset/binario/1id/00.jpg')
##plt.imshow(image)
#img_gray = rgb2gray(image)
#plt.imshow(image_gray, cmap=plt.cm.gray) 

#Most significant bit
image_blue = image[:,:,0]

image_blue_b = image_blue.ravel()

image_blue_b = image_blue >= 128

image_blue_b_lsb = image_blue & 1
