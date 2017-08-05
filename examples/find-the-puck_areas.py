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
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import optimizers
from keras.callbacks import TensorBoard

#Declares training/validation directories and filename format
dir_training    = '../find-the-puck/training/'
dir_validation  = '../find-the-puck/validation/'
filename_format = 'row_[XX]_col_[YY].jpg'
save_dir='../find-the-puck/images_generated/'

# Number of vertical/horizontal pixels
pixels_v = 120
pixels_h = 160
# Number of rows/columns to predict
rows     = 21
columns  = 32

# Number of training samples
num_training_samples = rows * columns
#Factor of incresing images
fact_gen=40
#Declares empty arrays for training inputs and outputs
inputs_train   = np.zeros( ( num_training_samples*(fact_gen+1), pixels_v, pixels_h,1) )
output_train = np.zeros( ( num_training_samples*(fact_gen+1), 9) )
#output_train_X = np.zeros( ( num_training_samples*(fact_gen+1), rows) )
#output_train_Y = np.zeros( ( num_training_samples*(fact_gen+1), columns) )

#Data generator
datagen = ImageDataGenerator(width_shift_range=0.05,height_shift_range=0.05,shear_range=0.05)

train_datafile='C:/Users/Jose Manuel/Documents/GitHub/datasets/examples/train_data_file.npy'
output_train_datafile='C:/Users/Jose Manuel/Documents/GitHub/datasets/examples/output_train_data_file.npy'
valid_datafile='C:/Users/Jose Manuel/Documents/GitHub/datasets/examples/valid_data_file.npy'
output_valid_datafile='C:/Users/Jose Manuel/Documents/GitHub/datasets/examples/output_valid_data_file.npy'


# Open files and generate images
if not os.path.isfile(train_datafile):
    counter=0
    for row in range(rows) :

        for col in range(columns) :

            filename = dir_training + filename_format
            filename = filename.replace( '[XX]', str(row).zfill(2) )
            filename = filename.replace( '[YY]', str(col).zfill(2) )

            #print( 'Loading training sample:', filename)
            print('Loading training sample row:%d, col %d' % (row+1, col+1))
            index = row * columns + col+fact_gen*counter

            im = Image.open(filename).convert('L')
            im.load()
            img_small = im.resize((pixels_h, pixels_v), Image.ANTIALIAS)
            inputs_train[index, :, :,0]=np.asarray(img_small)
            if col<=columns//3 and row <= rows//3:
                output_train[index,0]=1.0
            elif col>columns//3 and col <= columns*2//3 and row <= rows//3:
                output_train[index,1]=1.0
            elif col>columns*2//3 and row <= rows//3:
                output_train[index,2]=1.0
            elif col<=columns//3 and row > rows//3 and row <= rows*2//3:
                output_train[index,3]=1.0
            elif col>columns//3 and col <= columns*2//3  and row > rows//3 and row <= rows*2//3:
                output_train[index,4]=1.0
            elif col>columns*2//3  and row > rows//3 and row <= rows*2//3:
                output_train[index,5]=1.0
            elif col<=columns//3 and row > rows*2//3 :
                output_train[index,6]=1.0
            elif col>columns//3 and col <= columns*2//3  and row > rows*2//3 :
                output_train[index,7]=1.0
            elif col>columns*2//3  and row > rows*2//3 :
                output_train[index,8]=1.0
#            output_train_X[ index, row]   = 1.0
#            output_train_Y[ index, col]   = 1.0

            #Generate artificial training samples
            i=1
            im_np=np.array(im.getdata()).reshape((1,480,640,1))
            for batch in datagen.flow(im_np, batch_size=1):
                batch_2=batch.reshape(480,640)
                img_batch = Image.fromarray(batch_2)
                img_batch_small = img_batch.resize((pixels_h, pixels_v), Image.ANTIALIAS)
                inputs_train[ index+i, :, :,0]=np.asarray(img_batch_small)
                if col<=columns//3 and row <= rows//3:
                    output_train[index+i,0]=1.0
                elif col>columns//3 and col <= columns*2//3 and row <= rows//3:
                    output_train[index+i,1]=1.0
                elif col>columns*2//3 and row <= rows//3:
                    output_train[index+i,2]=1.0
                elif col<=columns//3 and row > rows//3 and row <= rows*2//3:
                    output_train[index+i,3]=1.0
                elif col>columns//3 and col <= columns*2//3  and row > rows//3 and row <= rows*2//3:
                    output_train[index+i,4]=1.0
                elif col>columns*2//3  and row > rows//3 and row <= rows*2//3:
                    output_train[index+i,5]=1.0
                elif col<=columns//3 and row > rows*2//3 :
                    output_train[index+i,6]=1.0
                elif col>columns//3 and col <= columns*2//3  and row > rows*2//3 :
                    output_train[index+i,7]=1.0
                elif col>columns*2//3  and row > rows*2//3 :
                    output_train[index+i,8]=1.0
#                output_train_X[ index+i, row]   = 1.0
#                output_train_Y[ index+i, col]   = 1.0
                i += 1
                if i > fact_gen:
                    break
            counter+=1
    np.save('train_data_file',inputs_train,allow_pickle=True)
    np.save('output_train_data_file',output_train,allow_pickle=True)
else:
    print('si hay archivo')
    inputs_train=np.load(train_datafile)
    output_train=np.load(output_train_datafile)

# Normalizes the training samples
#inputs_train_mean = np.mean( inputs_train, axis = 0)
#inputs_train_std  = np.std( inputs_train)
#inputs_train      = ( inputs_train - inputs_train_mean ) / inputs_train_std
#img_2=Image.fromarray(inputs_train[0,:,:,0])
#img_2.show()

# Extracts number of validation samples
validation_files = os.listdir('../find-the-puck/validation')
num_validation_samples = len(validation_files)

dir_validation  = '../find-the-puck/validation/'
filename_format = 'row_[XX]_col_[YY].jpg'


#Populates validation inputs and outputs
if not os.path.isfile(valid_datafile):
    inputs_valid   = np.zeros( ( num_validation_samples, pixels_v, pixels_h,1) )
    output_valid = np.zeros( ( num_validation_samples, 9) )
    #output_valid_X = np.zeros( ( num_validation_samples, rows) )
    #output_valid_Y = np.zeros( ( num_validation_samples, columns) )
    for ( index, filename) in enumerate(validation_files) :
        # Collects outputs
        row = int( filename[ 4: 6] )
        col = int( filename[-6:-4] )
        if col<=columns//3 and row <= rows//3:
            output_train[index,0]=1.0
        elif col>columns//3 and col <= columns*2//3 and row <= rows//3:
            output_valid[index,1]=1.0
        elif col>columns*2//3 and row <= rows//3:
            output_valid[index,2]=1.0
        elif col<=columns//3 and row > rows//3 and row <= rows*2//3:
            output_valid[index,3]=1.0
        elif col>columns//3 and col <= columns*2//3  and row > rows//3 and row <= rows*2//3:
            output_valid[index,4]=1.0
        elif col>columns*2//3  and row > rows//3 and row <= rows*2//3:
            output_valid[index,5]=1.0
        elif col<=columns//3 and row > rows*2//3 :
            output_valid[index,6]=1.0
        elif col>columns//3 and col <= columns*2//3  and row > rows*2//3 :
            output_valid[index,7]=1.0
        elif col>columns*2//3  and row > rows*2//3 :
            output_valid[index,8]=1.0
        # Collects inputs
        filename = dir_validation + filename
        print( 'Loading validation sample:', filename)
        im = Image.open(filename).convert('L')
        im.load()
        img_small = im.resize((pixels_h, pixels_v), Image.ANTIALIAS)

        inputs_valid[index, :, :,0]=np.asarray(img_small)
#        output_valid_X[ index, row]   = 1.0
#        output_valid_Y[ index, col]   = 1.0
    np.save('valid_data_file',inputs_valid,allow_pickle=True)
    np.save('output_valid_data_file',output_valid,allow_pickle=True)
else:
    inputs_valid=np.load(valid_datafile)
    output_valid=np.load(output_valid_datafile)

## Normalizes the validation samples
#inputs_valid_mean = np.mean( inputs_valid, axis = 0)
#inputs_valid_std  = np.std( inputs_valid)
#inputs_valid = ( inputs_valid - inputs_valid_mean ) / inputs_valid_std


sample_shape = ( pixels_v, pixels_h, 1)

layer_0 = Input( shape = sample_shape, name = 'Input_Images')

layer_11 = Conv2D( filters = 32, kernel_size = (5,5), strides = ( 2, 2),
                  name = '2D-Convolution',
                  activation = 'relu', data_format='channels_last' ,padding='same')( layer_0 )
layer_12 = Conv2D( filters = 64, kernel_size = (3,3), strides = ( 2, 2),
                  name = '2D-Convolution_2',
                  activation = 'relu' )( layer_11 )
layer_13= MaxPooling2D(pool_size=(2, 2))(layer_12)
#layer_14= Dropout(0.4)(layer_13)


layer_31 = Conv2D( filters = 64, kernel_size = (3,3), strides = ( 2, 2),
                  name = '2D-Convolution_3',
                  activation = 'relu' )( layer_13)
layer_32 = Conv2D( filters = 64, kernel_size = (3,3), strides = ( 2, 2),
                  name = '2D-Convolution_4',
                  activation = 'relu' )( layer_31)
layer_33= MaxPooling2D(pool_size=(2, 2))(layer_32)
#layer_34= Dropout(0.4)(layer_33)


layer_4 = Flatten( name = 'Flatten_Image-into-Vector' )( layer_33)
layer_5= Dense(256, activation='relu')(layer_4)
layer_6= Dropout(0.5)(layer_5)

row_output = Dense( units = 9, activation = 'softmax',
                    name = 'Row_Probabilities' )( layer_6 )
#col_output = Dense( units = columns, activation = 'softmax',
 #                   name = 'Col_Probabilities' )( layer_25 )

#neural_net = Model( inputs = layer_0, outputs = [ row_output, col_output])

neural_net = Model( inputs = layer_0, outputs = row_output)

neural_net.summary()
#losses = { 'Row_Probabilities' : 'categorical_crossentropy',
 #          'Col_Probabilities' : 'categorical_crossentropy' }
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
losses = { 'Row_Probabilities' : 'categorical_crossentropy'}
neural_net.compile( optimizer = optimizer, loss = losses, metrics = ['accuracy'] )

#plot_model( neural_net, to_file = 'model-architecture_2.png')

#neural_net.fit( x = inputs_train,
                #y = [ output_train_X, output_train_Y], epochs = 100,
                #validation_split=0.3)
callbacks=[TensorBoard(write_graph = False)]
neural_net.fit( inputs_train,
                output_train, epochs = 10,
                validation_split=0.2,shuffle=True,callbacks=callbacks)


pred_X = neural_net.predict( inputs_valid)