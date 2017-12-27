#!/usr/bin/env python3
"""
@author: Luis I. Reyes Castro
"""

# Number of pixels in the image across its row and col dimensions
ORIG_IMG_ROWS = 480
ORIG_IMG_COLS = 640

# Row and col limits (everything outside is irrelevant)
ORIG_IMG_ROW_LIM_1  = 60
ORIG_IMG_ROW_LIM_2  = 430
ORIG_IMG_COL_LIM_1  = 40
ORIG_IMG_COL_LIM_2  = 600

# Cropped image size (i.e. of the window of relevant features)
CROPPED_IMG_ROWS = ORIG_IMG_ROW_LIM_2 - ORIG_IMG_ROW_LIM_1
CROPPED_IMG_COLS = ORIG_IMG_COL_LIM_2 - ORIG_IMG_COL_LIM_1

# Names of ships, rows, cols and headings
NO_SHIP = 'Empty'
SHIPS   = [ 'Cruiser-1', 'Cruiser-2', 'Cruiser-3', 'Freighter',
           'Fishing-1', 'Fishing-2' ]
ROWS    = [ '1', '2', '3', '4', '5', '6', '7' ]
COLS    = [ 'A', 'B', 'C', 'D' ]
HEADS   = [ 'East', 'West' ]

# Numbers of rows, cols, headings and ships
NUM_SHIPS = len(SHIPS)
NUM_ROWS  = len(ROWS)
NUM_COLS  = len(COLS)
NUM_HEADS = len(HEADS)

# Indices where the token for no-ship starts and ends (if present)
IDX_NS_1 = 20
IDX_NS_2 = 25

# Indices where to find row, col, heading and ship (if present)
IDX_ROW    = 29
IDX_COL    = 30
IDX_HEAD_1 = 40
IDX_HEAD_2 = 44
IDX_SHIP_1 = 50
IDX_SHIP_2 = 59
