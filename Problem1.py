from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters

#Problem 1
#1.1,1.2 Utilizing a gaussian filter I blur the image and display its result
cheetah = misc.imread('cheetah.png',flatten=1)
cheetahold = misc.imread('cheetah.png',flatten=1)
blur = filters.gaussian_filter(cheetah,3)
plt.figure()
plt.imshow(blur,cmap=plt.cm.gray)
plt.show()
#Computing DFT
f=np.fft.fft2(cheetah)
plt.subplot(122),plt.imshow(cheetahold, cmap = 'gray')
plt.title('Display Old'), plt.xticks([]), plt.yticks([])
plt.show()








