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

# Declares training/validation directories and filename format
dir_training    = '../find-the-puck/training/'
dir_validation  = '../find-the-puck/validation/'
filename_format = 'row_[XX]_col_[YY].jpg'

# Number of vertical/horizontal pixels
pixels_v = 480
pixels_h = 640
# Number of rows/columns to predict
rows     = 21
columns  = 32
# Number of training samples
num_training_samples = rows * columns

# Declares empty arrays for training inputs and outputs
inputs_train   = np.zeros( ( num_training_samples, pixels_v, pixels_h, 1) )
output_train_X = np.zeros( ( num_training_samples, rows) )
output_train_Y = np.zeros( ( num_training_samples, columns) )

# Populates training inputs and outputs
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

# Normalizes the training samples
inputs_train_mean = np.mean( inputs_train, axis = 0)
inputs_train_std  = np.std( inputs_train)
inputs_train      = ( inputs_train - inputs_train_mean ) / inputs_train_std

# Extracts number of validation samples
validation_files = os.listdir('../find-the-puck/validation')
num_validation_samples = len(validation_files)

# Declares empty arrays for validation inputs and outputs
inputs_valid   = np.zeros( ( num_validation_samples, pixels_v, pixels_h, 1) )
output_valid_X = np.zeros( ( num_validation_samples, rows) )
output_valid_Y = np.zeros( ( num_validation_samples, columns) )

# Populates validation inputs and outputs
for ( index, filename) in enumerate(validation_files) :
    # Collects outputs
    row = int( filename[ 4: 6] )
    col = int( filename[-6:-4] )
    output_valid_X[ index, row] = 1.0
    output_valid_Y[ index, col] = 1.0
    # Collects inputs
    filename = dir_validation + filename
    print( 'Loading validation sample:', filename)
    inputs_valid[ index, :, :, 0] = imread( filename, flatten = True)

# Normalizes the validation samples
inputs_valid = ( inputs_valid - inputs_train_mean ) / inputs_train_std

sample_shape = ( pixels_v, pixels_h, 1)

layer_0 = Input( shape = sample_shape, name = 'Input_Images')

layer_1 = Conv2D( filters = 1, kernel_size = 150, strides = ( 20, 20),
                  name = '2D-Convolution',
                  activation = 'softsign' )( layer_0 )

layer_2 = Flatten( name = 'Flatten_Image-into-Vector' )( layer_1 )

row_output = Dense( units = rows, activation = 'softmax',
                    name = 'Row_Probabilities' )( layer_2 )
col_output = Dense( units = columns, activation = 'softmax',
                    name = 'Col_Probabilities' )( layer_2 )

neural_net = Model( inputs = layer_0, outputs = [ row_output, col_output])

losses = { 'Row_Probabilities' : 'categorical_crossentropy',
           'Col_Probabilities' : 'categorical_crossentropy' }

neural_net.compile( optimizer = 'sgd', loss = losses, metrics = ['accuracy'] )

plot_model( neural_net, to_file = 'model-architecture.png')

neural_net.fit( x = inputs_train,
                y = [ output_train_X, output_train_Y], epochs = 100,
                validation_data = ( inputs_valid, [ output_valid_X, output_valid_Y]) )
neural_net.save_weights('model-weights.h5')

#neural_net.load_weights('model-weights.h5')

( pred_X, pred_Y) = neural_net.predict( inputs_valid)
