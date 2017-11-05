import numpy as np
from datetime import datetime as dt
from cv2 import VideoCapture, imwrite

# Fraction of samples to be labeled and not-empty
fraction_nonempty = 0.8

# Rows, Columns, Headings and Ships
rows     = [ '1', '2', '3', '4', '5', '6', '7' ]
cols     = [ 'A', 'B', 'C', 'D' ]
headings = [ 'East', 'West' ]
ships    = [ 'Cruiser-1', 'Cruiser-2', 'Cruiser-3', 'Freighter',
             'Fishing-1', 'Fishing-2', 'Empty' ]

# Retrieves current date and time
filename = str( dt.today() )[:19]
filename = filename.replace( ' ', '_')
# Randomly selects row, col, heading and ship
r        = np.random.choice( rows)
c        = np.random.choice( cols)
h        = np.random.choice( headings)
s        = np.random.choice( ships)
filename = filename + '_Location:' + r + c \
         + '_Heading:' + h + '_Ship:' + s

# Prompts user to set up sample
print( 'Please generate sample at: ' )
print( 'LOCATION ' + r + c + ' HEADING ' + h + ' SHIP ' + s )

# Initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port. 
camera = VideoCapture(1)
key = raw_input( 'Are you ready? Press Enter to continue. ' )

# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 16
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
camera.release()
del(camera)