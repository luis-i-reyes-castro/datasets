# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:07:00 2017
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imshow
from keras.layers import Input, Flatten, Dense
from keras.layers import Conv2D
from keras.models import Model
from keras.utils.vis_utils import plot_model

rows    = 21
columns = 32
num_images = rows * columns
images_dir = '../vision-02/'
images_format = 'fila_[XX]_col_[YY].jpg'

inputs = np.zeros( shape = ( num_images, 480, 640, 1) )
output_XX = np.zeros( shape = ( num_images, rows) )
output_YY = np.zeros( shape = ( num_images, columns) )

for row in range(rows) :
    for col in range(columns) :
        filename = images_dir
        filename += images_format
        filename = filename.replace( '[XX]', str(row).zfill(2) )
        filename = filename.replace( '[YY]', str(col).zfill(2) )
        print( 'Reading image:', filename)
        image = imread( filename, flatten = True)
        image_index = row * columns + col
        inputs[ image_index, :, :, 0] = image
        output_XX[ image_index, row] = 1.0
        output_YY[ image_index, col] = 1.0

image_mean = np.mean( inputs, axis = 0)
image_std  = np.std( inputs, axis = 0)

inputs -= image_mean
inputs /= image_std

batch_shape = ( 32, 480, 640, 1)
layer_0 = Input( batch_shape = batch_shape, name = 'Input_images')
layer_1 = Conv2D( filters = 1, kernel_size = 150, strides = ( 10, 10),
                  name = '2D-Convolution',
                  activation = 'softsign')( layer_0)
layer_2 = Flatten()( layer_1)

row_output = Dense( units = rows, activation = 'softmax',
                    name = 'Row_predictions' )( layer_2)
col_output = Dense( units = columns, activation = 'softmax',
                    name = 'Col_predictions' )( layer_2)

neural_net = Model( inputs = layer_0, outputs = [ row_output, col_output])

losses = { 'Row_predictions' : 'categorical_crossentropy',
           'Col_predictions' : 'categorical_crossentropy' }
neural_net.compile( optimizer = 'sgd', loss = losses)

#neural_net.fit( x = inputs, y = [ output_XX, output_YY],
#                epochs = 10)
#neural_net.save_weights('prototipo-A.h5')

neural_net.load_weights('prototipo-A.h5')

indices_to_test = np.random.randint( num_images, size = 32)
inputs_to_test  = inputs[indices_to_test]

( pred_row, pred_col) = neural_net.predict( inputs_to_test)
plot_model( neural_net, to_file = 'Red-vision.png')
