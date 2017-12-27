#!/usr/bin/env python3
"""
@author: Luis I. Reyes Castro
"""

import os
import numpy as np
from scipy.misc import imread
from matplotlib.pyplot import figure, imshow
from PIL import Image
import dataset_constants as DSC

def get_filenames( directory) :

    if not os.path.exists(directory) :
        raise Exception( 'Could not find directory: ' + directory )

    return os.listdir(directory)

def print_dataset_stats( dataset_dir) :

    filenames   = get_filenames( dataset_dir)
    num_samples = len(filenames)
    fraction    = 100.0 / num_samples

    percent_ship = { ship : 0.0 for ship in DSC.SHIP }
    percent_rows = {  row : 0.0 for  row in DSC.ROWS + ['Empty'] }
    percent_cols = {  col : 0.0 for  col in DSC.COLS + ['Empty'] }
    percent_head = { head : 0.0 for head in DSC.HEAD + ['Empty'] }

    for ( index, filename) in enumerate(filenames) :

        if filename[ DSC.IDX_EMPTY_1 : DSC.IDX_EMPTY_2] == 'Empty' :
            percent_ship['Empty'] += fraction
            percent_rows['Empty'] += fraction
            percent_cols['Empty'] += fraction
            percent_head['Empty'] += fraction
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
    for ship in DSC.SHIP :
        print( '\t' + ship + ': ' + str(percent_ship[ship]) )
    print( 'ROWS:' )
    for row in DSC.ROWS + ['Empty'] :
        print( '\t' +  row + ': ' + str(percent_rows[row]) )
    print( 'COLS:' )
    for col in DSC.COLS + ['Empty'] :
        print( '\t' +  col + ': ' + str(percent_cols[col]) )
    print( 'HEADINGS:' )
    for head in DSC.HEAD + ['Empty'] :
        print( '\t' + head + ': ' + str(percent_head[head]) )

    return

def load_samples( dataset_dir) :

    filenames    = get_filenames( dataset_dir)
    num_samples  = len(filenames)
    input_tensor = np.zeros( ( num_samples, DSC.CROPPED_IMG_ROWS,
                                            DSC.CROPPED_IMG_COLS, 3) )

    for ( index, filename) in enumerate(filenames) :

        full_filename       = dataset_dir + filename
        print( 'Loading image:', full_filename)

        image_array         = imread( full_filename)
        input_tensor[index] = \
        image_array[ DSC.ORIG_IMG_ROW_LIM_1 : DSC.ORIG_IMG_ROW_LIM_2,
                     DSC.ORIG_IMG_COL_LIM_1 : DSC.ORIG_IMG_COL_LIM_2, :]

    return input_tensor

def show_random_sample( tensor, render_with = 'both') :

    num_samples = tensor.shape[0]
    index       = np.random.randint( num_samples)
    image_array = tensor[index]

    if render_with in [ 'matplotlib', 'both'] :
        figure( figsize = ( 10, 8) )
        imshow( image_array)

    if render_with in [ 'PIL', 'both'] :
        image = Image.fromarray( np.uint8(image_array) )
        image.show()

    return
