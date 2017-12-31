#!/usr/bin/env python3
"""
@author: Luis I. Reyes-Castro
"""

import numpy as np
from example_models import DetectorCNN

batch_size = 16
epochs     = 400
patience   = epochs // 10
workers    = 6

learning_rate       = [ 2E-5, 4E-5, 1E-4, 2E-4, 4E-4, 1E-3, 2E-3, 4E-3 ]
regularization_rate = [ 2E-6, 4E-6, 1E-5, 2E-5, 4E-5, 1E-4, 2E-4, 4E-4 ]
dropout_rate        = [ 0.15, 0.20, 0.25, 0.30 ]

while True :

    lr = np.random.choice( learning_rate)
    rr = np.random.choice( regularization_rate)
    dr = np.random.choice( dropout_rate)

    dcnn = DetectorCNN( batch_size, lr, rr, dr)
    dcnn.show_model()
    dcnn.train( epochs, patience, workers)
