#!/usr/bin/env python3
"""
@author: Luis I. Reyes Castro
"""

import os
from datetime import datetime as dtdt

import numpy as np
from scipy.misc import imread
from matplotlib.pyplot import figure, imshow
from PIL import Image
from pickle import dump, load, HIGHEST_PROTOCOL
from cv2 import VideoCapture, imwrite

import dataset_constants as DSC

def ensure_directory( directory) :

    directory += '/' if not directory[-1] == '/' else ''
    directory = os.path.dirname( directory + 'dummy-filename.txt' )

    if not os.path.exists( directory) :
        print( 'Did not find directory', directory)
        print( 'Creating directory:', directory)
        os.makedirs( directory)

    return

def exists_file( filename) :

    return os.path.exists( filename)

def get_filenames( directory) :

    if not os.path.exists(directory) :
        raise Exception( 'Could not find directory: ' + directory )

    return os.listdir(directory)

def get_todays_date() :

    return dtdt.now().strftime('%h-%d_%H:%M:%S')

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

def load_samples( dataset_dir, verbose = False) :

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

    print( 'LOADING SAMPLES...' )
    for ( index, filename) in enumerate(filenames) :

        full_filename      = dataset_dir + filename
        if verbose :
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

def serialize( obj_to_serialize, filename) :

    filename += '' if filename[-4:] == '.pkl' else '.pkl'

    handle = open( filename, 'wb')
    dump( obj_to_serialize, handle, HIGHEST_PROTOCOL)
    handle.close()

    return

def de_serialize( filename) :

    filename += '' if filename[-4:] == '.pkl' else '.pkl'

    handle = open( filename, 'rb')
    deserialized_obj = load( handle)
    handle.close()

    return deserialized_obj

def snap_photo( ramp_frames = 16) :

    filename = get_todays_date()
    filename += '.jpg'

    camera = VideoCapture(1)

    print( 'Ramping camera...' )
    for i in range(ramp_frames) :
        ( ret_val, image) = camera.read()

    print( 'Capturing and saving image:', filename)
    ret_val, image = camera.read()
    imwrite( filename, image)
    camera.release()

    input_array = np.zeros( shape = ( 1, DSC.ORIG_IMG_ROWS, DSC.ORIG_IMG_COLS, 1),
                            dtype = np.float32 )
    input_array[ 0, :, :, 0] = imread( filename, flatten = True)

    return input_array
