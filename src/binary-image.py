#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 08:43:38 2017

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

image0 = imread( '/home/luis/datasets/vision-01/1id/00.jpg', flatten = True)
image1 = imread( '/home/luis/datasets/vision-01/1id/01.jpg', flatten = True)

todas = np.stack( [ image0, image1])

imagen_prom = np.mean( todas, axis = 0)
imagen_std  = np.std( todas)

todas_preprocesadas = ( todas - imagen_prom ) / imagen_std
plt.imshow( todas_preprocesadas[0,:,:] )
