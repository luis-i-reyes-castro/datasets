# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:07:00 2017
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

images = []

for root, dirnames, filenames in os.walk("/home/user/Documents/Investigacion/IA_Image/datasets/vision-02/"):    
    for filename in filenames:
        filepath = os.path.join(root, filename)
        image = imread(filepath, flatten = True)
        images.append(image)
        todas = np.stack( [images])
    imagen_prom = np.mean( todas, axis = 0)
    imagen_std  = np.std( todas)
    todas_preprocesadas = ( todas - imagen_prom ) / imagen_std
    #plt.imshow( todas_preprocesadas[0,:,:] )
