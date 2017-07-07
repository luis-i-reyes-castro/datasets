#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:47:42 2017

@author: josemanuel
"""

import numpy as np
import os
from scipy.misc import imread, imshow
from keras.layers import Input, Flatten, Dense, MaxPooling2D, Dropout
from keras.layers import Conv2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import optimizers

#Declares training/validation directories and filename format
dir_training    = '../find-the-puck/training/'
dir_validation  = '../find-the-puck/validation/'
filename_format = 'row_[XX]_col_[YY].jpg'
save_dir='../find-the-puck/images_generated/'

# Number of vertical/horizontal pixels
pixels_v = 240
pixels_h = 320
# Number of rows/columns to predict
rows     = 21
columns  = 32

# Number of training samples
num_training_samples = rows * columns
#Factor of incresing images
fact_gen=5
#Declares empty arrays for training inputs and outputs
inputs_train   = np.zeros( ( num_training_samples*fact_gen+1, pixels_v, pixels_h, 1) )
output_train_X = np.zeros( ( num_training_samples*fact_gen+1, rows) )
output_train_Y = np.zeros( ( num_training_samples*fact_gen+1, columns) )

#Data generator
datagen = ImageDataGenerator(zca_whitening=True, shear_range=0.2,fill_mode='nearest')

# Open files and generate images
for row in range(rows) :

    for col in range(columns) :
        
        filename = dir_training + filename_format
        filename = filename.replace( '[XX]', str(row).zfill(2) )
        filename = filename.replace( '[YY]', str(col).zfill(2) )
        
        #print( 'Loading training sample:', filename)
        print('Loading training sample row:%d, col %d' % (row, col))
        index = row * columns + col
        
        im = Image.open(filename).convert('L')
        img_small = im.resize((pixels_h, pixels_v), Image.ANTIALIAS)
        
        inputs_train[index, :, :, :]=np.array(img_small.getdata()).reshape(pixels_v,pixels_h,1)
        output_train_X[ index, row]   = 1.0
        output_train_Y[ index, col]   = 1.0

        #Generate artificial training samples
        i=1
        im_np=np.array(im.getdata()).reshape(1,480,640,1)
        for batch in datagen.flow(im_np, batch_size=1):
            batch_2=batch.reshape(640,480)
            img_batch = Image.fromarray(batch_2, mode='L')
            img_batch_small = img_batch.resize((pixels_h, pixels_v), Image.ANTIALIAS)
            inputs_train[ index+i, :, :, :]=np.array(img_batch_small.getdata()).reshape(pixels_v,pixels_h,1)
            output_train_X[ index+i, row]   = 1.0
            output_train_Y[ index+i, col]   = 1.0
            i += 1
            if i > fact_gen:
                break 
            
## Normalizes the training samples
inputs_train_mean = np.mean( inputs_train, axis = 0)
inputs_train_std  = np.std( inputs_train)
inputs_train      = ( inputs_train - inputs_train_mean ) / inputs_train_std      

# Extracts number of validation samples
validation_files = os.listdir('../find-the-puck/validation')
num_validation_samples = len(validation_files)

dir_validation  = '../find-the-puck/validation/'
filename_format = 'row_[XX]_col_[YY].jpg'
#Declares empty arrays for validation inputs and outputs
inputs_valid   = np.zeros( ( num_validation_samples, pixels_v, pixels_h, 1) )
output_valid_X = np.zeros( ( num_validation_samples, rows) )
output_valid_Y = np.zeros( ( num_validation_samples, columns) )
#Populates validation inputs and outputs
for ( index, filename) in enumerate(validation_files) :
    # Collects outputs
    row = int( filename[ 4: 6] )
    col = int( filename[-6:-4] )
    output_valid_X[ index, row] = 1.0
    output_valid_Y[ index, col] = 1.0
    # Collects inputs
    filename = dir_validation + filename
    print( 'Loading validation sample:', filename)
    im = Image.open(filename).convert('L')
    img_small = im.resize((pixels_h, pixels_v), Image.ANTIALIAS)
        
    inputs_valid[index, :, :, :]=np.array(img_small.getdata()).reshape(pixels_v,pixels_h,1)
    output_valid_X[ index, row]   = 1.0
    output_valid_Y[ index, col]   = 1.0    
    

## Normalizes the validation samples
inputs_valid_mean = np.mean( inputs_valid, axis = 0)
inputs_valid_std  = np.std( inputs_valid)
inputs_valid = ( inputs_valid - inputs_valid_mean ) / inputs_valid_std


sample_shape = ( pixels_v, pixels_h, 1)

layer_0 = Input( shape = sample_shape, name = 'Input_Images')

layer_1 = Conv2D( filters = 20, kernel_size = (5,5), strides = ( 2, 2),
                  name = '2D-Convolution',
                  activation = 'softsign' )( layer_0 )
layer_11= Dropout(0.3)(layer_1)

layer_12 = Conv2D( filters = 50, kernel_size = (5,5), strides = ( 2, 2),
                  name = '2D-Convolution_2',
                  activation = 'softsign' )( layer_11 )

layer_13= MaxPooling2D(pool_size=(2, 2))(layer_12)

#layer_14 = Conv2D( filters = 100, kernel_size = (5,5), strides = ( 2, 2),
#                  name = '2D-Convolution_3',
#                  activation = 'softsign' )( layer_13)


layer_2 = Flatten( name = 'Flatten_Image-into-Vector' )( layer_13)

layer_22= Dense(100, activation='relu')(layer_2)

layer_23= Dropout(0.3)(layer_22)

#layer_24= Dense(100, activation='relu')(layer_23)

#layer_25= Dense(100, activation='relu')(layer_23)

row_output = Dense( units = rows, activation = 'softmax',
                    name = 'Row_Probabilities' )( layer_23 )
#col_output = Dense( units = columns, activation = 'softmax',
 #                   name = 'Col_Probabilities' )( layer_25 )

#neural_net = Model( inputs = layer_0, outputs = [ row_output, col_output])

neural_net = Model( inputs = layer_0, outputs = row_output)

neural_net.summary()
#losses = { 'Row_Probabilities' : 'categorical_crossentropy',
 #          'Col_Probabilities' : 'categorical_crossentropy' }

losses = { 'Row_Probabilities' : 'categorical_crossentropy'}
neural_net.compile( optimizer = 'sgd', loss = losses, metrics = ['accuracy'] )

plot_model( neural_net, to_file = 'model-architecture_2.png')

#neural_net.fit( x = inputs_train,
                #y = [ output_train_X, output_train_Y], epochs = 100,
                #validation_split=0.3)
neural_net.fit( inputs_train,
                output_train_X, epochs = 100,
                validation_split=0.3,shuffle=True)


pred_X = neural_net.predict( inputs_valid)