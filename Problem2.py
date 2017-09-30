#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Adonay
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import exposure

#Problem 2 
#i believe setting true below should set the image to grayscale
#if not i could also try to use 
zebra = ndimage.imread('zebra.png',False,'L')
hist,bins = np.histogram(zebra.flatten(),256,[0,256])
#found on stack overflow
#norm.cdf(zebra)
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(zebra.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

#Most white pure white most black pure black top 5 %
