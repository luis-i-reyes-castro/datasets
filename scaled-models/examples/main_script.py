#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import utilities as util

util.print_dataset_stats( '../set-A_train/')
util.print_dataset_stats( '../set-B_test/')

tensor_A = util.load_samples( '../set-A_train/')
tensor_B = util.load_samples( '../set-B_test/')
