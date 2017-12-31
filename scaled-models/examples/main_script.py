#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import numpy as np
import utilities as utils
from example_models import DetectorCNN

batch_size = 16
epochs     = 500

dcnn = DetectorCNN( batch_size)
dcnn.show_model()
dcnn.train( batch_size, epochs)
