#!/usr/bin/env python3
"""
@author: Luis I. Reyes Castro
"""

import os
import numpy as np
from scipy.misc import imread
from matplotlib.pyplot import figure, imshow
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

import dataset_constants as DSC

def get_filenames( directory) :

    if not os.path.exists(directory) :
        raise Exception( 'Could not find directory: ' + directory )

    return os.listdir(directory)

def print_dataset_stats( dataset_dir) :

    filenames   = get_filenames( dataset_dir)
    num_samples = len(filenames)
    fraction    = 100.0 / num_samples

    ships = DSC.SHIPS + ['Empty']
    rows  = DSC.ROWS + ['Empty']
    cols  = DSC.COLS + ['Empty']
    heads = DSC.HEADS + ['Empty']

    percent_ship = { ship : 0.0 for ship in ships }
    percent_rows = {  row : 0.0 for  row in  rows }
    percent_cols = {  col : 0.0 for  col in  cols }
    percent_head = { head : 0.0 for head in heads }

    for ( index, filename) in enumerate(filenames) :

        if filename[ DSC.IDX_NS_1 : DSC.IDX_NS_2] == DSC.NO_SHIP :
            percent_ship[ DSC.NO_SHIP ] += fraction
            percent_rows[ DSC.NO_SHIP ] += fraction
            percent_cols[ DSC.NO_SHIP ] += fraction
            percent_head[ DSC.NO_SHIP ] += fraction
        else :
            ship = filename[ DSC.IDX_SHIP_1 : DSC.IDX_SHIP_2 ]
            row  = filename[ DSC.IDX_ROW ]
            col  = filename[ DSC.IDX_COL ]
            head = filename[ DSC.IDX_HEAD_1 : DSC.IDX_HEAD_2 ]
            percent_ship[ship] += fraction
            percent_rows[ row] += fraction
            percent_cols[ col] += fraction
            percent_head[head] += fraction

    print( 'ANALYSIS OF DATASET:', dataset_dir)
    print( 'SHIPS:' )
    for ship in ships :
        print( '\t' + ship + ': ' + str(percent_ship[ship]) )
    print( 'ROWS:' )
    for  row in  rows :
        print( '\t' +  row + ': ' + str(percent_rows[row]) )
    print( 'COLS:' )
    for  col in  cols :
        print( '\t' +  col + ': ' + str(percent_cols[col]) )
    print( 'HEADINGS:' )
    for head in heads :
        print( '\t' + head + ': ' + str(percent_head[head]) )

    return

def load_samples( dataset_dir) :

    filenames   = get_filenames( dataset_dir)
    num_samples = len(filenames)

    input_array = np.zeros( ( num_samples, DSC.ORIG_IMG_ROWS,
                                           DSC.ORIG_IMG_COLS, 1),
                            dtype = np.float32)

    out_array_prob = np.zeros( shape = ( num_samples, 1), dtype = np.float32)
    out_array_ship = np.zeros( shape = ( num_samples, DSC.NUM_SHIPS),
                               dtype = np.float32)
    out_array_row  = np.zeros( shape = ( num_samples, DSC.NUM_ROWS ),
                               dtype = np.float32)
    out_array_col  = np.zeros( shape = ( num_samples, DSC.NUM_COLS ),
                               dtype = np.float32)
    out_array_head = np.zeros( shape = ( num_samples, DSC.NUM_HEADS),
                               dtype = np.float32)

    for ( index, filename) in enumerate(filenames) :

        full_filename      = dataset_dir + filename
        print( 'Loading image:', full_filename)
        input_array[ index, :, :, 0] = imread( full_filename, flatten = True)

        if not filename[ DSC.IDX_NS_1 : DSC.IDX_NS_2] == DSC.NO_SHIP :

            ship = filename[ DSC.IDX_SHIP_1 : DSC.IDX_SHIP_2 ]
            row  = filename[ DSC.IDX_ROW ]
            col  = filename[ DSC.IDX_COL ]
            head = filename[ DSC.IDX_HEAD_1 : DSC.IDX_HEAD_2 ]

            out_array_prob[ index, 0] = 1.0
            out_array_ship[ index, DSC.SHIPS.index(ship) ] = 1.0
            out_array_row[ index, DSC.ROWS.index(row) ]    = 1.0
            out_array_col[ index, DSC.COLS.index(col) ]    = 1.0
            out_array_head[ index, DSC.HEADS.index(head) ] = 1.0

    output_array = np.concatenate( ( out_array_prob,
                                     out_array_ship,
                                     out_array_row,
                                     out_array_col,
                                     out_array_head), axis = -1)

    return ( input_array, output_array)

def batch_generator( X_tensor, Y_tensor, batch_size = 32) :

    generator = ImageDataGenerator( rotation_range = DSC.IDG_ROTATION,
                                    zoom_range = DSC.IDG_ZOOM,
                                    width_shift_range = DSC.IDG_WIDTH_SHIFT,
                                    height_shift_range = DSC.IDG_HEIGHT_SHIFT,
                                    fill_mode = DSC.IDG_FILL_MODE )

    def cropped_img_batch_generator() :

        cropped_x_batch = \
        np.zeros( shape = ( batch_size, DSC.CROPPED_IMG_ROWS,
                                        DSC.CROPPED_IMG_COLS, 1),
                  dtype = np.float32 )

        for ( x_batch, y_batch) in generator.flow( X_tensor, Y_tensor,
                                                   batch_size = batch_size) :
            cropped_x_batch[:,:,:,:] = \
            x_batch[ :, DSC.ORIG_IMG_ROW_LIM_1 : DSC.ORIG_IMG_ROW_LIM_2,
                        DSC.ORIG_IMG_COL_LIM_1 : DSC.ORIG_IMG_COL_LIM_2, :]

            yield ( cropped_x_batch, y_batch)

        return

    return cropped_img_batch_generator()

def show_random_sample( tensor, render_with = 'matplotlib') :

    num_samples = tensor.shape[0]
    index       = np.random.randint( num_samples)
    image_array = tensor[ index, :, :, 0]

    if render_with in [ 'matplotlib', 'both'] :
        figure( figsize = ( 10, 8) )
        imshow( image_array )

    if render_with in [ 'PIL', 'both'] :
        image = Image.fromarray( np.uint8(image_array) )
        image.show()

    return
