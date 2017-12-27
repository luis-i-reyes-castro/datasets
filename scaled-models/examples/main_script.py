#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import utilities as util

util.print_dataset_stats( '../set-A_train/')
util.print_dataset_stats( '../set-B_test/')

( input_A, output_A) = util.load_samples( '../set-A_train/')
( input_B, output_B) = util.load_samples( '../set-A_train/')
