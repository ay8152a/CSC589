# -*- coding: utf-8 -*-


import numpy as np
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage

img = misc.imread('apple.jpg',flatten=1)
imgo= misc.imread('orange.jpg',flatten=1)

# create a  Binomial (5-tap) filter
kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])

plt.imshow(kernel)
plt.show()
#img_up = np.zeros((2*img.shape[0], 2*img.shape[1]))
#img_up[::2, ::2] = img
#ndimage.filters.convolve(img_up,4*kernel, mode='constant')

#sig.convolve2d(img_up, 4*kernel, 'same')

def interpolate(image):
    """
    Interpolates an image with upsampling rate r=2.
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    # Upsample
    image_up[::2, ::2] = image
    # Blur (we need to scale this up since the kernel has unit area)
    # (The length and width are both doubled, so the area is quadrupled)
    #return sig.convolve2d(image_up, 4*kernel, 'same')
    return ndimage.filters.convolve(image_up,4*kernel, mode='constant')
                                
def decimate(image):
    """
    Decimates at image with downsampling rate r=2.
    """
    # Blur
    #image_blur = sig.convolve2d(image, kernel, 'same')
    image_blur = ndimage.filters.convolve(image,kernel, mode='constant')
    # Downsample
    return image_blur[::2, ::2]                                
               
    
# here is the constructions of pyramids
def pyramids(image):
    """
    Constructs Gaussian and Laplacian pyramids.
    Parameters :
        image  : the original image (i.e. base of the pyramid)
    Returns :
        G   : the Gaussian pyramid
        L   : the Laplacian pyramid
    """
    # Initialize pyramids
    G = [image, ]
    L = []

    # Build the Gaussian pyramid to maximum depth
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)
   
    # Build the Laplacian pyramid
    for i in range(len(G) - 1):
         L.append(G[i] - interpolate(G[i + 1]))
       
    
    return G[:-1], L
                                
#interpolate(img)
#decimate(img)
[G,L] = pyramids(img)



# reconstruct the pyramids, here you write a reconstrut function that takes the 
# pyramid and upsampling the each level and add them up.    
def reconstruct(L,G):
   
    img1=L[-1]
    b=len(G)-1
    for i in range(len(G) - 1):
        img2 = np.zeros((2*img1.shape[0], 2*img1.shape[1]))
        img2 = img2[::1, ::1]
        g=b-i
        img1 = G[g]+img2
        b-=1
    return img1
         #K = []
        #g=0
    #for i in L[::-1]:
       # img2 = np.zeros((2*L[i].shape[0], 2*L[i].shape[1]))
       # img2 = img2[::2, ::2] 
       # K.append(L[i]+img2)
        #g+=1
def blender(image1,image2):
    S=[]
    mask= misc.imread('mask.jpg',flatten=1)
    [A,B] = pyramids(image1)
    [C,D] = pyramids(image2)
    [E,F] = pyramids(mask)
    j = len(E)
    mask = ndimage.filters.convolve(mask,4*kernel, mode='constant')
    for i in range(0,j):
        b1= mask*B[i]
        b2= (1-mask)*D[i]
        S.append(b1+b2)
    f=reconstruct(b1,b2) #if my reconstruct code worked i would utilize it for the blend code to collapse the pyramid to get the blended image 
    return(f)




rows, cols = img.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
composite_image[:rows, :cols] = G[0]

i_row = 0
for p in G[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


fig, ax = plt.subplots()
    
ax.imshow(composite_image,cmap='gray')
plt.show()


rows, cols = img.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)

composite_image[:rows, :cols] = L[0]

i_row = 0
for p in L[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


fig, ax = plt.subplots()
    
ax.imshow(composite_image,cmap='gray')
plt.show()

f = reconstruct(L,G)
c= blender(img,imgo)
ax.imshow(f,cmap='gray')
plt.show()
                                   