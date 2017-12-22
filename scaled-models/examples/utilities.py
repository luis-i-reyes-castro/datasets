#!/usr/bin/env python3
"""
@author: Luis I. Reyes Castro
"""

import os
import numpy as np
from scipy.misc import imread
from matplotlib.pyplot import figure, imshow
from PIL import Image

PIXEL_ROWS = 480
PIXEL_COLS = 640
PIX_LIM_U  = 50
PIX_LIM_D  = 430
PIX_LIM_L  = 40
PIX_LIM_R  = 600

NUM_ROWS = 7
NUM_COLS = 4
NUM_HEAD = 2
NUM_SHIP = 6

data_dir = '../set-A_train/'

if not os.path.exists(data_dir) :
    raise Exception( 'Could not find directory!' )

filenames   = os.listdir(data_dir)
num_samples = len(filenames)

i_tensor = np.zeros( ( num_samples, PIXEL_ROWS, PIXEL_COLS, 3) )

for ( index, filename) in enumerate(filenames) :
    full_filename = data_dir + filename
    print( 'Loading image:', full_filename)
    i_tensor[ index, :, :, :] = imread( full_filename)

def random_image( render_with = 'matplotlib') :

    index = np.random.randint( num_samples )
    array = i_tensor[ index, PIX_LIM_U : PIX_LIM_D, PIX_LIM_L : PIX_LIM_R, :]

    if render_with == 'matplotlib' :
        figure( figsize = ( 10, 8) )
        imshow( array)
    elif render_with == 'PIL' :
        image = Image.fromarray( np.uint8(array) )
        image.show()

    return array
