import numpy as np
from datetime import datetime as dt
from cv2 import VideoCapture, imwrite

# Fraction of samples to be labeled and not-empty
fraction_labeled  = 0.8
fraction_nonempty = 0.8
# Ships, headings and their respective fractions (probabilities)
ships   = [ 'Cruiser-A', 'Cruiser-B', 'Cruiser-C', 'Freighter', 'Fishing' ]
heading = [ 'East', 'West' ]
fraction_ships   = [ 1./6, 1./6, 1./6, 1./6, 1./3 ]
fraction_heading = [ 0.5, 0.5 ]

# Initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port. 
camera = VideoCapture(0)
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
filename = str( dt.today() )[:19] + '.jpg'
filename = filename.replace( ' ', '_')
print( 'Writing image: ' + filename )
imwrite( filename, image)

# Release the camera so that other threads can use it later
camera.release()
del(camera)