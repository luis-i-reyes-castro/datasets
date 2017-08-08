#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:18:50 2017

@author: josemanuel
"""
import numpy as np
import os
from scipy.misc import imread, imshow
from keras.layers import Input, Flatten, Dense
from keras.layers import Conv2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

#Declares training/validation directories and filename format
dir_training    = '../find-the-puck/training/'
dir_validation  = '../find-the-puck/validation/'
filename_format = 'row_[XX]_col_[YY].jpg'
filename_new_format = 'row_[XX]_col_[YY]_artificial'
save_dir='../find-the-puck/images_generated/'

# Number of vertical/horizontal pixels
pixels_v = 480
pixels_h = 640
# Number of rows/columns to predict
rows     = 21
columns  = 32


# Declares empty arrays for training inputs and outputs
datagen = ImageDataGenerator(shear_range=0.2,fill_mode='nearest')

# Open files and generate images
for row in range(rows) :

    for col in range(columns) :
        
        filename = dir_training + filename_format
        filename_new=filename_new_format
        filename = filename.replace( '[XX]', str(row).zfill(2) )
        filename = filename.replace( '[YY]', str(col).zfill(2) )
        filename_new = filename_new.replace( '[XX]', str(row).zfill(2) )
        filename_new = filename_new.replace( '[YY]', str(col).zfill(2) )
        
        print( 'Loading training sample:', filename)
        index = row * columns + col
        
        im = Image.open(filename).convert('L')
        img = im.resize((pixels_h, pixels_v), Image.ANTIALIAS)
        imagen=np.array(img.getdata()).reshape(1,pixels_v,pixels_h,1)
        #datagen.fit(imagen)
        i=0
        for batch in datagen.flow(imagen, batch_size=1, save_to_dir=save_dir, save_prefix=filename_new, save_format='jpeg'):
            i += 1
            if i > 2:
                break  # otherwise the generator would loop indefinitely
            break
        break
    break
        
    