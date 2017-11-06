import numpy as np
from datetime import datetime as dt
from cv2 import VideoCapture, imwrite

# Rows, Columns, Headings and Ships
rows     = [ '1', '2', '3', '4', '5', '6', '7' ]
cols     = [ 'A', 'B', 'C', 'D' ]
headings = [ 'East', 'West' ]
ships    = [ 'Cruiser-1', 'Cruiser-2', 'Cruiser-3', 'Freighter',
             'Fishing-1', 'Fishing-2', 'Empty' ]
# Directory names for the training and test sets
dataset_dirs = [ 'set-A_train/', 'set-B_test/' ]
# Fraction of samples for the training and test sets
fraction_train = 0.8
# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 16

# Loop indefinitely...
while True :

    # Retrieves current date and time
    filename = str( dt.today() )[:19]
    filename = filename.replace( ' ', '_')
    
    # Randomly selects ship, row, column and heading
    ship    = np.random.choice( ships)
    row     = np.random.choice( rows)
    col     = np.random.choice( cols)
    heading = np.random.choice( headings)
    
    # Prompts user to set up sample
    print( ' ' )
    print( 'Please setup: ' )
    # If no ship (i.e. clear waters)
    if ship == ships[-1] :
        filename = filename + '_' + ships[-1]
        print( 'EMPTY IMAGE' )    
    # Else, if requested to sample a ship
    else :
        filename = filename \
        + '_Location:' + row + col + '_Heading:' + heading + '_Ship:' + ship
        print( 'SHIP: ' + ship )
        print( 'HEADING: ' + heading )
        print( 'LOCATION: ' + row + col )
    
    # Assign the approriate directory to the output file
    if np.random.rand() < fraction_train :
        filename = dataset_dirs[0] + filename
    else :
        filename = dataset_dirs[1] + filename
    
    # Initialize the camera capture object with the cv2.VideoCapture class.
    # All it needs is the index to a camera port. 
    camera = VideoCapture(1)
    key = raw_input( 'When ready, press Enter to capture image.' )
    
    # Ramp the camera - these frames will be discarded and are only used 
    # to allow v4l2 to adjust light levels, if necessary
    print( 'Ramping camera...' )
    for i in xrange( ramp_frames) :
        return_val, image = camera.read()
    
    # Take the actual image we want to keep
    print( 'Capturing actual image...' )
    return_val, image = camera.read()
    
    # Save the image in JPG format
    filename += '.jpg'
    print( 'Writing image: ' + filename)
    imwrite( filename, image)
    
    # Release the camera so that other threads can use it later
    print( 'Done!' )
    camera.release()
    del(camera)
