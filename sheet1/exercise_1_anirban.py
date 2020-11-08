#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:01:50 2020

@author: anirban
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 

img = cv.imread("brain.png")

#a)

ret, binary_img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
cv.imwrite("brain_bin.png", binary_img)
#plt.show()


#2)
gradient = np.linspace(0, 1, num=255)
image = np.tile(gradient, (100, 1))
#print(image)
#image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.imshow(image, cmap='gray')
#plt.savefig("gradient.png", image, cmap="gray")
plt.show()
#cv.imwrite("gradient.png", image)
