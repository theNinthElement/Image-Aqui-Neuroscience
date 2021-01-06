#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:10:13 2021

@author: anirban
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import scipy.stats

def to_image(x):
    x = x - np.min(x)
    x = x / np.max(x) * 255
    x = np.uint8(x)
    return x

#a)
img = cv2.imread('brain-noisy.png', 0)
plt.imshow(img, cmap='gray')
plt.show()

img2 = cv2.imread('mask.png', 0)
plt.imshow(img2, cmap='gray')
plt.show()

def pot(fi, fj):
    return float((fi-fj))**2

def ICM(img):
    
    h, w = img.shape
    sigma2 = 5
    beta = 0.5
    
    iterations = 5
    
    for i in range(iterations):
        print("iteration {}\n".format(i+1))
        for j in range(h-1):
            print("line {}/{} ok\n".format(j+1, h))
            for k in range(w-1):
                xmin = 0
                min = float((img[j][k]*img[j][k]))/(2.0*sigma2) + \
                beta*(pot(img[j][k-1],0)+pot(img[j][k+1],0)+pot(img[j-1][k], 0)+pot(img[j+1][k], 0))
                    
                for x in range(256):
                    prob = float(((img[j][k]-x)*(img[j][k]-x)))/(2.0*sigma2) + \
                    beta*(pot(img[j][k-1],x) + pot(img[j][k+1],x) + pot(img[j-1][k], x) + pot(img[j+1][k], x))
                    if(min>prob):
                        min = prob
                        xmin = x
                img [j][k] = xmin    
    return img
    
final_img = ICM(img)

plt.imshow(final_img, cmap='gray')
plt.show()

