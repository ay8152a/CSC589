#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: Adonay
"""
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters
from scipy.stats import norm
from scipy import ndimage

#Problem 3
zebra = filters.ndimage.imread('lowcontrast.jpg',False,'L')
ndimage.convolve(zebra,output=zebra)


