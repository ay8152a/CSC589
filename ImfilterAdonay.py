#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:02:22 2017

@author: Gengar
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
import cv2
 


#Loading in the images 
#pice = cv2.imread('bird.bmp',1)
pice= ndimage.imread('bird.bmp',False,'L')
picm = ndimage.imread('plane.bmp',False,'L')
#red, green, blue = cv2.split(pice)
#red = pice[:,:,2] 
#blue = pice[:,:,0] = 0 
#green = pice[:,:,1] = 0
#plt.imshow(blue)
#plt.show()
 
    
#blurring the images using gaussian filter 
gpice = ndimage.gaussian_filter(pice, 2)
# I effectively use this gaussian filter of picm my plane image as lowpass
gpicm = ndimage.gaussian_filter(picm, 3)
plt.imshow(gpice,cmap='gray')
plt.show()
plt.imshow(gpicm,cmap='gray')
plt.show()



#removing low freq by subtracting gaussian blurred from original image array

#first attempt to remove low frequcencys however it did not work 
#kernel = np.array([[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]])
#highpass = ndimage.convolve(hp, kernel)
#highpass = gpice - pice
scale =2
delta =0
kernel_size=3
ddepth=cv2.CV_16S
highpass = cv2.Laplacian(pice,ddepth,ksize = kernel_size,scale = scale,delta = delta)


hp = ndimage.gaussian_filter(highpass, 1)


#combining low pass and high pass images to create hybrid image 
hp = hp + gpicm


plt.imshow(hp,cmap='gray')

