#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import numpy as np
import utilities as util
import dataset_constants as DSC
from keras.preprocessing.image import ImageDataGenerator

( input_array, outputs) = util.load_samples( '../set-B_test/')

#generator = ImageDataGenerator( rotation_range = 2,
#                                zoom_range = 0.02,
#                                width_shift_range = 0.02,
#                                height_shift_range = 0.02,
#                                fill_mode = 'constant',
#                                featurewise_center=True,
#                                featurewise_std_normalization=True)
#generator.fit( input_array)
#
#for ( x_batch, y_batch) in generator.flow( input_array, outputs[0],
#                                           batch_size = 64) :
#    break
