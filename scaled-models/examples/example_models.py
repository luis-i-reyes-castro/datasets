#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

import utilities as utils
import dataset_constants as DSC

class DetectorCNN :

    MODEL_DIR    = 'trained-models/'
    MODEL_ID     = 'model-[TIME]_LR-[LR]_REG-[REG]_DP-[DP]'
    TB_LOG_DIR   = 'tensorboard-logs/'

    def __init__( self, batch_size = 32,
                        learning_rate = 1E-3,
                        regularization_rate = 1E-4,
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

        layer_3A = Conv2D( filters = 128,
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
                                 pool_size = (6,8) )( layer_3B )

        layer_FL = Flatten( name = 'Flatten-into-Vector')( layer_MP )

        layer_D1 = Dense( units = 512, activation = 'softsign',
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

        self.model = Model( inputs = layer_00, outputs = output_tensors)

        optimizer = SGD( lr = learning_rate,
                         momentum = 0.9, nesterov = True )

        self.model.compile( optimizer = optimizer,
                            loss = loss,
                            loss_weights = loss_weights,
                            metrics = ['accuracy'] )

        return

    def show_model( self, filename = 'Model-Architecture.jpg') :

        self.model.summary()

        return plot_model( model = self.model,
                           show_shapes = True, to_file = filename)

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
