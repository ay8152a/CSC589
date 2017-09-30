from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters
from scipy.stats import norm
#Problem 1
#1.1,1.2 Utilizing a gaussian filter I blur the image and display its result
cheetah = misc.imread('cheetah.png',flatten=1)
blur = filters.gaussian_filter(cheetah,3)
plt.figure()
plt.imshow(blur,cmap=plt.cm.gray)
plt.show()
#Computing DFT
f=np.fft.fft2(cheetah)
#plt.subplot(100),plt.imshow(f, cmap = 'gray')
#plt.title('Display mag'), plt.xticks([]), plt.yticks([])
#plt.show()


#img = cv2.imread('wiki.jpg',0)
#equ = cv2.equalizeHist(img)
#res = np.hstack((img,equ)) #stacking images side-by-side
#cv2.imwrite('res.png',res)
# number 4 and five uncompleted

#Problem 3
zebra = filters.ndimage.imread('lowcontrast.jpg',False,'L')
ndimage.convolve(zebra,output=zebra)






