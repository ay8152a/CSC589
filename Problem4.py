#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Adonay
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import ndimage

edge = misc.imread('raury.jpg')
edge = edge.astype('int32')
#part of this was gathered from stack overflow I did not understand how to use derivatives for sobel
dx = ndimage.sobel(edge, 0)  # horizontal derivative
dy = ndimage.sobel(edge, 1)  # vertical derivative
mag = np.hypot(dx, dy)  # magnitude
mag *= 255.0 / np.max(mag)  # normalize (Q&D)
misc.imsave('sobel.jpg', mag)
plt.imshow(edge,cmap=plt.cm.gray)
plt.show()
#The image will be saved to this folder and will be visible after runnning this code.