#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:49:24 2020

@author: anirban
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
import scipy.ndimage as ndimage
import scipy.misc as misc 


def parta(img):
    
    new_translated_img = ndimage.rotate(img, -30)
    new_translated_img = ndimage.shift(new_translated_img, np.array([60,2,0]))
    
    plt.imshow(img)
    plt.show()
    plt.imshow(new_translated_img)
    #plt.axes().set_aspect('equal', 'datalim')
    plt.show()  
    #plt.imshow(img, cmap='gray')
    #plt.show()
    return 0

def L2Norm(img1, img2):
    cost = 0    
    cost = np.sum((img1-img2)^2)    
    return cost

def partb(img):
    
    translate_left = ndimage.zoom(ndimage.shift(img, np.array([0,50,0])), (1,1,1))
    translate_right = ndimage.zoom(ndimage.shift(img, np.array([50,0,0])), (1,1,1))
    rotated =  ndimage.rotate(img, -45)
    
    print("translate x ", L2Norm(img,translate_left))
    print("translate y ", L2Norm(img,translate_right))
    #print("rotation " , L2Norm(img,rotated))
    #print(L2Norm(img,rotated_left1))
    
    return 0



if __name__ == "__main__":
    path = "axial.png"
    transalted_path = "axial_transformed.png"
    #img = misc.face()
    #translated_img = misc.face()
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    translated_img = cv.imread(transalted_path)
    translated_img = cv.cvtColor(translated_img, cv.COLOR_BGR2RGB)
    #img = Image.fromarray(img)
    #parta(img)
    partb(img)
    

