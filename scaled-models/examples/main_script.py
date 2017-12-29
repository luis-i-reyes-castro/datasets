#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import numpy as np
import utilities as util
import dataset_constants as DSC

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

reg_level    =  1E-4
dropout_rate = 0.2

sample_shape = ( DSC.CROPPED_IMG_ROWS, DSC.CROPPED_IMG_COLS, 1)

layer_0 = Input( shape = sample_shape, name = 'Input_Images')

layer_1A = Conv2D( filters = 32, kernel_size = (3,3), strides = (2,2),
                   name = 'Convolution-1',
                   activation = 'relu',
                   data_format = 'channels_last',
                   padding = 'same',
                   kernel_regularizer = l2(reg_level) )( layer_0 )
layer_1B = Dropout( rate = dropout_rate, name = 'Dropout-1' )( layer_1A )

layer_2A = Conv2D( filters = 64, kernel_size = (3,3), strides = (2,2),
                   name = 'Convolution-2',
                   activation = 'relu',
                   data_format = 'channels_last',
                   padding = 'same',
                   kernel_regularizer = l2(reg_level) )( layer_1B )
layer_2B = Dropout( rate = dropout_rate, name = 'Dropout-2' )( layer_2A )

layer_3A = Conv2D( filters = 128, kernel_size = (3,3), strides = (2,2),
                   name = 'Convolution-3',
                   activation = 'relu',
                   data_format = 'channels_last',
                   padding = 'same',
                   kernel_regularizer = l2(reg_level) )( layer_2B )
layer_3B = Dropout( rate = dropout_rate, name = 'Dropout-3' )( layer_3A )

layer_3B = MaxPooling2D( name = 'Max-pooling',
                         pool_size = (6,8) )( layer_3B )

layer_FLAT = Flatten( name = 'Flatten-into-vec')( layer_3B )

layer_FC_1 = Dense( units = 512, activation = 'softsign',
                    name = 'Fully-Connected',
                    kernel_regularizer = l2(reg_level) )( layer_FLAT )

output_prob = Dense( units = 1, activation = 'sigmoid',
                     name = 'Output_Ship-Prob',
                     kernel_regularizer = l2(reg_level) )( layer_FC_1 )
output_ship = Dense( units = DSC.NUM_SHIPS, activation = 'softmax',
                     name = 'Output_Ship',
                     kernel_regularizer = l2(reg_level) )( layer_FC_1 )
output_row  = Dense( units = DSC.NUM_ROWS, activation = 'softmax',
                     name = 'Output_Row',
                     kernel_regularizer = l2(reg_level) )( layer_FC_1 )
output_col  = Dense( units = DSC.NUM_COLS, activation = 'softmax',
                     name = 'Output_Col',
                     kernel_regularizer = l2(reg_level) )( layer_FC_1 )
output_head = Dense( units = DSC.NUM_HEADS, activation = 'softmax',
                     name = 'Output_Head',
                     kernel_regularizer = l2(reg_level) )( layer_FC_1 )

loss = { 'Output_Ship-Prob' : 'binary_crossentropy',
         'Output_Ship'      : 'categorical_crossentropy',
         'Output_Row'       : 'categorical_crossentropy',
         'Output_Col'       : 'categorical_crossentropy',
         'Output_Head'      : 'categorical_crossentropy' }

loss_weights = { 'Output_Ship-Prob' : 8.0,
                 'Output_Ship'      : 1.0,
                 'Output_Row'       : 2.0,
                 'Output_Col'       : 2.0,
                 'Output_Head'      : 1.0 }

model = Model( inputs = layer_0, outputs = [ output_prob,
                                             output_ship,
                                             output_row,
                                             output_col,
                                             output_head ] )

model.compile( optimizer = 'sgd',
               loss = loss,
               loss_weights = loss_weights,
               metrics = ['accuracy'] )

model.summary()
plot_model(model)

( X_tensor, Y_tensor) = util.load_samples( '../set-A_train/')

X_tensor -= X_tensor.mean()
X_tensor /= X_tensor.std()

num_samples = X_tensor.shape[0]
batch_size = 16
epochs = 100
patience = 20
steps_per_epoch = num_samples // batch_size

set_A_gen = util.batch_generator( X_tensor, Y_tensor, batch_size)

model.fit_generator( generator = set_A_gen, epochs = epochs,
                     steps_per_epoch = steps_per_epoch,
                     use_multiprocessing = True,
                     workers = 4)
