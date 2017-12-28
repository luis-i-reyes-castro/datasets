#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import utilities as util
import dataset_constants as DSC

( input_array, outputs) = util.load_samples( '../set-A_train/')

generator = ImageDataGenerator( rotation_range = 1.5,
                                zoom_range = 0.02,
                                width_shift_range = 0.02,
                                height_shift_range = 0.02,
                                fill_mode = 'constant')

input_array -= input_array.mean()
input_array /= input_array.std()

for ( x_batch, y_batch) in generator.flow( input_array, outputs[0],
                                           batch_size = 64) :
    x_batch = \
    x_batch[:, DSC.ORIG_IMG_ROW_LIM_1 : DSC.ORIG_IMG_ROW_LIM_2,
               DSC.ORIG_IMG_COL_LIM_1 : DSC.ORIG_IMG_COL_LIM_2, :]
    break
