"""
@author: Luis I. Reyes-Castro
"""

import numpy as np
import os
from scipy.misc import imread, imshow
from keras.layers import Input, Flatten, Dense
from keras.layers import Conv2D
from keras.models import Model
from keras.utils.vis_utils import plot_model

dir_training    = '../find-the-puck/training/'
dir_validation  = '../find-the-puck/validation/'
filename_format = 'row_[XX]_col_[YY].jpg'

pixels_v = 480
pixels_h = 640
rows     = 21
columns  = 32
num_training_samples = rows * columns

inputs_train   = np.zeros( ( num_training_samples, pixels_v, pixels_h, 1) )
output_train_X = np.zeros( ( num_training_samples, rows) )
output_train_Y = np.zeros( ( num_training_samples, columns) )

for row in range(rows) :

    for col in range(columns) :

        filename = dir_training + filename_format
        filename = filename.replace( '[XX]', str(row).zfill(2) )
        filename = filename.replace( '[YY]', str(col).zfill(2) )

        print( 'Loading training sample:', filename)
        index = row * columns + col
        inputs_train[ index, :, :, 0] = imread( filename, flatten = True)
        output_train_X[ index, row]   = 1.0
        output_train_Y[ index, col]   = 1.0

inputs_train_mean = np.mean( inputs_train, axis = 0)
inputs_train_std  = np.std( inputs_train, axis = 0)
inputs_train = ( inputs_train - inputs_train_mean ) / inputs_train_std

validation_files = os.listdir('../find-the-puck/validation')
num_validation_samples = len(validation_files)

inputs_valid   = np.zeros( ( num_validation_samples, pixels_v, pixels_h, 1) )
output_valid_X = np.zeros( ( num_validation_samples, rows) )
output_valid_Y = np.zeros( ( num_validation_samples, columns) )

for ( index, filename) in enumerate(validation_files) :

    filename = dir_validation + filename
    print( 'Loading validation sample:', filename)

    inputs_valid[ index, :, :, 0] = imread( filename, flatten = True)

    row = int( filename[ 4: 6] )
    col = int( filename[-6:-4] )
    break

    output_valid_X[ index, row] = 1.0
    output_valid_Y[ index, col] = 1.0

sample_shape = ( pixels_v, pixels_h, 1)
layer_0 = Input( shape = sample_shape, name = 'Normalized_Images')
layer_1 = Conv2D( filters = 1, kernel_size = 150, strides = ( 30, 30),
                  name = '2D-Convolution',
                  activation = 'softsign')( layer_0 )
layer_2 = Flatten( name = 'Flatten_Image-into-Vector' )( layer_1 )

row_output = Dense( units = rows, activation = 'softmax',
                    name = 'Row_Probabilities' )( layer_2 )
col_output = Dense( units = columns, activation = 'softmax',
                    name = 'Col_Probabilities' )( layer_2 )

neural_net = Model( inputs = layer_0, outputs = [ row_output, col_output])

losses = { 'Row_Probabilities' : 'categorical_crossentropy',
           'Col_Probabilities' : 'categorical_crossentropy' }
neural_net.compile( optimizer = 'sgd', loss = losses)
plot_model( neural_net, to_file = 'model-architecture.png')

neural_net.fit( x = inputs_train,
                y = [ output_train_X, output_train_Y], epochs = 10)
neural_net.save_weights('model-weights.h5')
neural_net.load_weights('model-weights.h5')
