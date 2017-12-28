#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import utilities as util
import dataset_constants as DSC

( X_tensor, Y_tensor) = util.load_samples( '../set-A_train/')

X_tensor -= X_tensor.mean()
X_tensor /= X_tensor.std()

generator = util.batch_generator( X_tensor, Y_tensor, 64)

for ( x_batch, y_batch) in generator :
    break
