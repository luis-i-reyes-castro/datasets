#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import numpy as np

import utilities as util
from example_models import DetectorCNN


( X_train, Y_train) = util.load_samples( '../set-A_train/' )
( X_test,  Y_test)  = util.load_samples( '../set-B_test/' )

X_train_mean = X_train.mean()
X_train_std  = X_train.std()

X_train -= X_train_mean
X_train /= X_train_std
X_test  -= X_train_mean
X_test  /= X_train_std

batch_size = 16
epochs     = 100

steps_train = X_train.shape[0] // batch_size
steps_test  = X_test.shape[0]  // batch_size

dcnn = DetectorCNN( batch_size)

set_A_gen = dcnn.batch_generator( X_train, Y_train, batch_size)
set_B_gen = dcnn.batch_generator( X_test,  Y_test,  batch_size)

dcnn.model.fit_generator( generator = set_A_gen,
                          steps_per_epoch = steps_train,
                          validation_data = set_B_gen,
                          validation_steps = steps_test,
                          epochs = epochs,
                          use_multiprocessing = True,
                          workers = 4 )
