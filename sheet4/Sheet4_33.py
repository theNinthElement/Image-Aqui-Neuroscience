# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:23:44 2021

@author: Tobias
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

def to_image(x):
    x = x - np.min(x)
    x = x / np.max(x) * 255
    x = np.uint8(x)
    return x

"""Exercise 3 (Markov Random Fields)"""
#a)
img = cv2.imread('brain-noisy.png', 0)
plt.imshow(img, cmap='gray')
plt.show()
mask = cv2.imread('mask.png', 0)
mask = mask > 0
non_zero_pixels = img[mask]

#since our solution for exercise 2 doesn't seem to give the correct results, we use k-means as initialization
def init_kmeans(x, k):
    x_ = x.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k).fit(x_)
    mu = kmeans.cluster_centers_
    mu_ = mu[:, 0]
    nn = kmeans.predict(x_)
    sigma = np.empty(k)
    for i in range(k):
        sigma[i] = np.sqrt(np.mean((x[nn == i] - mu_[i])**2))
    return mu, sigma, nn

mu, sigma, kmeans = init_kmeans(non_zero_pixels, 3)

def visualize_labels_rgb(mask, labels):
    out_gray = np.zeros((mask.shape[0], mask.shape[1]))
    out_gray[mask] = labels+1
    out_rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                out_rgb[i, j, int(out_gray[i, j]-1)] = 255
    plt.imshow(to_image(out_rgb))
    plt.show()

visualize_labels_rgb(mask, kmeans)

#b)
def icm(x, labels, mask, beta, mu, sigma):
    n = x.shape[0]
        
    #calculate unary potentials
    unary = np.empty((3, n))
    for k in range(3):
        unary[k] = np.log(np.sqrt(2*np.pi) * sigma[k]) + (x - mu[k])**2 / (2 * sigma[k])
    
    #calculate binary potentials
    binary = np.empty((3, n, 4))
    (x,y) = np.where(mask)
    for j in range(3):
        for i in range(n):
            i1 = x[i]
            i2 = y[i]
            ind = np.logical_and(x == i1-1, y == i2)
            neighborhood = np.empty(4, dtype=np.int)
            if np.any(ind):
                neighborhood[0] = labels[np.where(ind)[0]]
            ind = np.logical_and(x == i1, y == i2-1)
            if np.any(ind):
                neighborhood[1] = labels[np.where(ind)[0]]
            ind = np.logical_and(x == i1, y == i2+1)
            if np.any(ind):
                neighborhood[2] = labels[np.where(ind)[0]]
            ind = np.logical_and(x == i1+1, y == i2+1)
            if np.any(ind):
                neighborhood[3] = labels[np.where(ind)[0]]
            binary[j, i] = beta * (neighborhood != j).astype(int)
    
    #assign new labels
    new_labels = np.argmin(unary + np.sum(binary, axis=2), axis=0)
        
    return new_labels

def get_mu_sigma(x, labels):
    mu = np.empty(3)
    sigma = np.empty(3)
    for i in range(3):
        clusterpoints = x[labels == i]
        mu[i] = np.mean(clusterpoints)
        sigma[i] = np.sqrt(np.mean((clusterpoints - mu[i])**2))
    return mu, sigma

old_labels = icm(non_zero_pixels, kmeans, mask, 0.5, mu, sigma)
visualize_labels_rgb(mask, old_labels)

#c)
for _ in range(5):
    new_labels = icm(non_zero_pixels, old_labels, mask, 0.5, mu, sigma)
    print("Number of changes:", np.sum(new_labels != old_labels))
    old_labels = new_labels
    mu, sigma = get_mu_sigma(non_zero_pixels, new_labels)
    visualize_labels_rgb(mask, new_labels)
