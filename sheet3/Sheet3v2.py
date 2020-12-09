# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:04:13 2020

@author: Tobias
"""

import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
parser = argparse.ArgumentParser(description='Sheet 3')
parser.add_argument("--image", type=str, default="axial.png")
parser.add_argument("--image_transformed", type=str, default="axial_transformed.png")
args = parser.parse_args()

def translate_rotate(img, rotation, shift):
    return nd.shift(nd.rotate(img[:], rotation), shift)

def L2Norm(fixed_img, moving_img):
    img2 = moving_img[:]
    if moving_img.shape != fixed_img.shape:
        diff_x = fixed_img.shape[0] - moving_img.shape[0]
        diff_y = fixed_img.shape[1] - moving_img.shape[1]
        img2 = np.pad(img2, [(math.ceil(diff_x/2), math.floor(diff_x/2)), (math.ceil(diff_y/2), math.floor(diff_y/2))], mode="constant", constant_values=0)
        if img2.shape != fixed_img.shape: print("FAIL")
    return np.sum((fixed_img-img2)**2) / np.size(fixed_img)

def find_initial_bracket(fixed_img, moving_img):
    bound = max(fixed_img.shape)
    a_shift = np.random.randint(-bound, bound)
    a_rotation = np.random.randint(360)
    a_l2 = L2Norm(translate_rotate(moving_img, a_rotation, a_shift), fixed_img)
    b_shift = np.random.randint(-bound, bound)
    b_rotation = np.random.randint(360)
    b_l2 = L2Norm(translate_rotate(moving_img, b_rotation, b_shift), fixed_img)
    
    l = 0.62 / 0.38
    c_shift = l * (b_shift - a_shift) + a_shift
    c_rotation = l * (b_rotation - a_rotation) + a_rotation
    c_l2 = L2Norm(translate_rotate(moving_img, c_rotation, c_shift), fixed_img)
    if a_shift < b_shift < c_shift and a_rotation < b_rotation < c_rotation and a_l2 > b_l2 and c_l2 > b_l2:
        return [(a_shift, a_rotation, a_l2), (b_shift, b_rotation, b_l2), (c_shift, c_rotation, c_l2)]
    else: return find_initial_bracket(fixed_img, moving_img) #try again
        
def refine_bracket(bracket, fixed_img, moving_img, precision):
    converged = False
    while not converged:
        print(bracket)
        #optimize for translation first
        a = bracket[0][0]
        b = bracket[1][0]
        c = bracket[2][0]
        if c-b < b-a:
            x_shift = a + 0.38*(b-a)
            x_rotation = (bracket[0][1]+bracket[1][1]) / 2 #take the average for the rotation
            x_l2 = L2Norm(translate_rotate(moving_img, x_rotation, x_shift), fixed_img) 
            x = (x_shift, x_rotation, x_l2)
            if x_l2 < bracket[1][2]: #choose (b,x,c)
                bracket[0] = bracket[1]
                bracket[1] = x
            else: #choose (a,b,x)
                bracket[2] = x
        else:
            x_shift = b + 0.38*(c-b)
            x_rotation = (bracket[1][1]+bracket[2][1]) / 2 #take the average for the rotation
            x_l2 = L2Norm(translate_rotate(moving_img, x_rotation, x_shift), fixed_img)
            x = (x_shift, x_rotation, x_l2)
            if x_l2 < bracket[1][2]: #choose (a,x,b)
                bracket[2] = bracket[1]
                bracket[1] = x
            else: #choose (x,b,c)
                bracket[0] = x
        
        #now optimize for rotation
        a = bracket[0][1]
        b = bracket[1][1]
        c = bracket[2][1]
        if c-b < b-a:
            x_rotation = a + 0.38*(b-a)
            x_shift = (bracket[0][0]+bracket[1][0]) / 2 #take the average for the shift
            x_l2 = L2Norm(translate_rotate(moving_img, x_rotation, x_shift), fixed_img) 
            x = (x_shift, x_rotation, x_l2)
            if x_l2 < bracket[1][2]: #choose (b,x,c)
                bracket[0] = bracket[1]
                bracket[1] = x
            else: #choose (a,b,x)
                bracket[2] = x
        else:
            x_rotation = b + 0.38*(c-b)
            x_shift = (bracket[1][0]+bracket[2][0]) / 2 #take the average for the shift
            x_l2 = L2Norm(translate_rotate(moving_img, x_rotation, x_shift), fixed_img)
            x = (x_shift, x_rotation, x_l2)
            if x_l2 < bracket[1][2]: #choose (a,x,b)
                bracket[2] = bracket[1]
                bracket[1] = x
            else: #choose (x,b,c)
                bracket[0] = x
        
        converged = bracket[2][0] - bracket[0][0] < precision and bracket[2][1] - bracket[0][1] < precision
    return bracket
    
    
fixed_img = plt.imread(args.image)
moving_img = plt.imread(args.image_transformed)
print(fixed_img.shape, moving_img.shape)
print(L2Norm(fixed_img, moving_img))
initial_bracket = find_initial_bracket(fixed_img, moving_img)
print("initial bracket found")
print(refine_bracket(initial_bracket, fixed_img, moving_img, 10))