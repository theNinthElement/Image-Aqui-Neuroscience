# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:39:42 2020

@author: Tobias
"""

import numpy as np
import matplotlib.pyplot as plt
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
import argparse
parser = argparse.ArgumentParser(description='Sheet 2')
parser.add_argument("--image_file", type=str, default="brain.png")
parser.add_argument("--epi", type=str, default="sepi.npy")
args = parser.parse_args()

def to_image(x):
    x = x - np.min(x)
    x = x / np.max(x) * 255
    x = np.uint8(x)
    return x

img = plt.imread(args.image_file)

#1.a)
plt.imshow(img, cmap='gray')
plt.show()
img_ = np.fft.fft2(img)
img_ = np.fft.fftshift(img_) #intermediate result we need for task b
img = np.log(np.abs(img_)**2)
img = to_image(img)
img[img < 125] = 125 #optional
img = to_image(img)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.imshow(img, cmap='gray')
plt.show()

#1.b)
"""
The result won't change, because the inverse fourier transform calculates a weighted sum over the frequencies in the k-space.
Thus, if we add frequency-entries with zero-values (formally, we add frequencies u with F(u)=0) the weighted sum doesn't change.

"""
x = np.pad(img_, [(500, 500), (500, 500)], mode="constant", constant_values=0)
x = np.fft.ifftshift(x)
x = np.fft.ifft2(x)
x = to_image(x)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.imshow(x, cmap='gray')
plt.show()
"""
We can't recnonstruct an image from the power spectrum alone, because the time-information is missing,
i.e. we know which frequencies occur in the image but not where. The time information is implicitly encoded in the complex numbers. 
"""

#1.c)
arr = np.load(args.epi)
arr = np.reshape(arr, (64, 64))
for i in range(arr.shape[0]):
    if i % 2 == 1:
        arr[i] = np.flip(arr[i], 0)
img = np.fft.ifft2(arr)
img = np.abs(img)
img = to_image(img)
img = np.fft.fftshift(img)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.imshow(img, cmap='gray')
plt.show()