#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro

COPYRIGHT

All contributions by Luis I. Reyes-Castro:
Copyright (c) 2018, Luis Ignacio Reyes Castro.
All rights reserved.
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import utilities as utils
import dataset_constants as DSC

class DetectorCNN :

    MODEL_DIR    = 'trained-models/'
    MODEL_ID     = 'model-[TIME]_LR-[LR]_REG-[REG]_DP-[DP]'
    TB_LOG_DIR   = 'tensorboard-logs/'

    def __init__( self, batch_size = 16,
                        learning_rate = 1E-4,
                        regularization_rate = 1E-5,
                        dropout_rate = 0.2 ) :

        self.batch_size          = batch_size
        self.learning_rate       = learning_rate
        self.regularization_rate = regularization_rate
        self.dropout_rate        = dropout_rate

        timestamp     = utils.get_todays_date()
        self.MODEL_ID = self.MODEL_ID.replace( '[TIME]', timestamp)
        self.MODEL_ID = self.MODEL_ID.replace( '[LR]',
                                               str(learning_rate) )
        self.MODEL_ID = self.MODEL_ID.replace( '[REG]',
                                               str(regularization_rate) )
        self.MODEL_ID = self.MODEL_ID.replace( '[DP]',
                                               str(dropout_rate) )

        self.model_definition = self.MODEL_DIR + self.MODEL_ID + '.pkl'
        self.model_weights    = self.MODEL_DIR + self.MODEL_ID + '_weights.h5'
        self.tb_log_dir       = self.TB_LOG_DIR + self.MODEL_ID

        input_shape = ( DSC.CROPPED_IMG_ROWS, DSC.CROPPED_IMG_COLS, 1)

        layer_00 = Input( shape = input_shape, name = 'Input_Images')

        layer_1A = Conv2D( filters = 32,
                           kernel_size = (3,3), strides = (2,2),
                           name = 'Convolution-1',
                           activation = 'relu',
                           data_format = 'channels_last',
                           padding = 'same',
                           kernel_regularizer = \
                           l2( regularization_rate) )( layer_00 )

        layer_1B = Dropout( rate = dropout_rate,
                            name = 'Dropout-1' )( layer_1A )

        layer_2A = Conv2D( filters = 64,
                           kernel_size = (3,3), strides = (2,2),
                           name = 'Convolution-2',
                           activation = 'relu',
                           data_format = 'channels_last',
                           padding = 'same',
                           kernel_regularizer = \
                           l2( regularization_rate) )( layer_1B )

        layer_2B = Dropout( rate = dropout_rate,
                            name = 'Dropout-2' )( layer_2A )

        layer_3A = Conv2D( filters = 32,
                           kernel_size = (3,3), strides = (2,2),
                           name = 'Convolution-3',
                           activation = 'relu',
                           data_format = 'channels_last',
                           padding = 'same',
                           kernel_regularizer = \
                           l2( regularization_rate) )( layer_2B )

        layer_3B = Dropout( rate = dropout_rate,
                            name = 'Dropout-3' )( layer_3A )

        layer_MP = MaxPooling2D( name = 'Max-pooling',
                                 pool_size = (2,4) )( layer_3B )

        layer_FL = Flatten( name = 'Flatten-into-Vector')( layer_MP )

        layer_D1 = Dense( units = 1024, activation = 'softsign',
                          name = 'Fully-Connected',
                          kernel_regularizer = \
                          l2( regularization_rate) )( layer_FL )

        output_prob = Dense( units = 1, activation = 'sigmoid',
                             name = 'Output_Ship-Prob',
                             kernel_regularizer = \
                             l2( regularization_rate) )( layer_D1 )

        output_ship = Dense( units = DSC.NUM_SHIPS, activation = 'softmax',
                             name = 'Output_Ship',
                             kernel_regularizer = \
                             l2( regularization_rate) )( layer_D1 )

        output_row  = Dense( units = DSC.NUM_ROWS, activation = 'softmax',
                             name = 'Output_Row',
                             kernel_regularizer = \
                             l2( regularization_rate) )( layer_D1 )

        output_col  = Dense( units = DSC.NUM_COLS, activation = 'softmax',
                             name = 'Output_Col',
                             kernel_regularizer = \
                             l2( regularization_rate) )( layer_D1 )

        output_head = Dense( units = DSC.NUM_HEADS, activation = 'softmax',
                             name = 'Output_Head',
                             kernel_regularizer = \
                             l2( regularization_rate) )( layer_D1 )

        output_tensors = [ output_prob, output_ship,
                           output_row, output_col, output_head ]

        self.model = Model( inputs = layer_00, outputs = output_tensors)

        self.loss = { 'Output_Ship-Prob' : 'binary_crossentropy',
                      'Output_Ship'      : 'categorical_crossentropy',
                      'Output_Row'       : 'categorical_crossentropy',
                      'Output_Col'       : 'categorical_crossentropy',
                      'Output_Head'      : 'categorical_crossentropy' }

        self.loss_weights = { 'Output_Ship-Prob' : 8.0,
                              'Output_Ship'      : 1.0,
                              'Output_Row'       : 2.0,
                              'Output_Col'       : 2.0,
                              'Output_Head'      : 1.0 }

        self.optimizer = SGD( lr = learning_rate,
                              momentum = 0.9, nesterov = True )

        return

    def show_model( self, filename = 'Model-Architecture.jpg') :

        self.model.summary()

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)

    def save_model( self) :

        print( 'Saving DetectorCNN model definition to file:',
               self.model_definition )

        model_dict = { 'batch_size' : self.batch_size,
                       'learning_rate' : self.learning_rate,
                       'regularization_rate' : self.regularization_rate,
                       'dropout_rate' : self.dropout_rate,
                       'model_definition' : self.model_definition,
                       'model_weights' : self.model_weights,
                       'model_tb_log' : self.tb_log_dir }

        utils.ensure_directory( self.MODEL_DIR)
        utils.serialize( model_dict, self.model_definition)

        return

    @staticmethod
    def load_model( model_definition_file) :

        if not utils.exists_file( model_definition_file) :
            raise ValueError( 'Did not find file:',
                              str(model_definition_file) )

        model_def = utils.de_serialize( model_definition_file)

        dcnn = DetectorCNN( model_def['batch_size'],
                            model_def['learning_rate'],
                            model_def['regularization_rate'],
                            model_def['dropout_rate'] )

        dcnn.model.load_weights( model_def['model_weights'] )

        return dcnn

    def train( self, batch_size = 16,
                     epochs = 400,
                     patience = None,
                     workers = 4,
                     augment_validation_data = True,
                     monitor = 'val_Output_Row_acc' ) :

        self.save_model()
        self.model.compile( optimizer = self.optimizer,
                            loss = self.loss,
                            loss_weights = self.loss_weights,
                            metrics = ['accuracy'] )

        call_mc = ModelCheckpoint( filepath = self.model_weights,
                                   monitor = monitor,
                                   save_weights_only = True,
                                   save_best_only = True )

        utils.ensure_directory( self.tb_log_dir)
        call_tb = TensorBoard( log_dir = self.tb_log_dir,
                               write_graph = False )

        callbacks = [ call_mc, call_tb]

        if patience is not None and isinstance( patience, int) :
            call_es = EarlyStopping( monitor = monitor,
                                     patience = patience )
            callbacks.append( call_es)

        ( X_t, Y_t) = utils.load_samples( DSC.SET_A_DIR)
        ( X_v, Y_v) = utils.load_samples( DSC.SET_B_DIR)

        self.X_mean = X_t.mean()
        X_t -= self.X_mean
        X_v -= self.X_mean

        self.X_std = X_t.std()
        X_t /= self.X_std
        X_v /= self.X_std

        t_data = self.batch_generator( X_t, Y_t, batch_size)

        if augment_validation_data :
            v_data = self.batch_generator( X_v,  Y_v,  batch_size)
        else :
            v_data = ( X_v,  Y_v)

        t_steps = X_t.shape[0] // batch_size
        v_steps = X_v.shape[0] // batch_size

        self.model.fit_generator( generator = t_data,
                                  steps_per_epoch = t_steps,
                                  validation_data = v_data,
                                  validation_steps = v_steps,
                                  epochs = epochs,
                                  use_multiprocessing = True,
                                  workers = workers,
                                  callbacks = callbacks )

        return

    def batch_generator( self, X_tensor, Y_tensor, batch_size = 32) :

        gen_ = ImageDataGenerator( rotation_range = DSC.IDG_ROTATION,
                                   zoom_range = DSC.IDG_ZOOM,
                                   width_shift_range = DSC.IDG_WIDTH_SHIFT,
                                   height_shift_range = DSC.IDG_HEIGHT_SHIFT,
                                   fill_mode = DSC.IDG_FILL_MODE )

        generator = gen_.flow( X_tensor, Y_tensor,
                               batch_size = self.batch_size )

        def cropped_img_batch_generator() :

            for ( x_batch, y_batch) in generator :

                cropped_x_batch = \
                x_batch[ :, DSC.ORIG_IMG_ROW_LIM_1 : DSC.ORIG_IMG_ROW_LIM_2,
                            DSC.ORIG_IMG_COL_LIM_1 : DSC.ORIG_IMG_COL_LIM_2, :]

                y_batch_prob = y_batch[ :, 0 : 1 ]
                y_batch_ship = \
                y_batch[ :, DSC.OUT_IDX_SHIP_1 : DSC.OUT_IDX_SHIP_2 ]
                y_batch_row  = \
                y_batch[ :,  DSC.OUT_IDX_ROW_1 :  DSC.OUT_IDX_ROW_2 ]
                y_batch_col  = \
                y_batch[ :,  DSC.OUT_IDX_COL_1 :  DSC.OUT_IDX_COL_2 ]
                y_batch_head = \
                y_batch[ :, DSC.OUT_IDX_HEAD_1 : DSC.OUT_IDX_HEAD_2 ]


                yield ( cropped_x_batch, [ y_batch_prob,
                                           y_batch_ship,
                                           y_batch_row,
                                           y_batch_col,
                                           y_batch_head ] )

            return

        return cropped_img_batch_generator()
