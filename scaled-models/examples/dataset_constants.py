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

# Image Data Generator (IDG) configuration
IDG_ROTATION     = 1.5
IDG_ZOOM         = 0.02
IDG_WIDTH_SHIFT  = 0.02
IDG_HEIGHT_SHIFT = 0.02
IDG_FILL_MODE    = 'constant'

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

# Indices in the output array where to find ship probability,
# ship, row, col and heading
OUT_IDX_PROB   = 0
OUT_IDX_SHIP_1 = 1
OUT_IDX_SHIP_2 = OUT_IDX_SHIP_1 + NUM_SHIPS
OUT_IDX_ROW_1  = OUT_IDX_SHIP_2
OUT_IDX_ROW_2  = OUT_IDX_ROW_1 + NUM_ROWS
OUT_IDX_COL_1  = OUT_IDX_ROW_2
OUT_IDX_COL_2  = OUT_IDX_COL_1 + NUM_COLS
OUT_IDX_HEAD_1 = OUT_IDX_COL_2
OUT_IDX_HEAD_2 = OUT_IDX_HEAD_1 + NUM_HEADS
