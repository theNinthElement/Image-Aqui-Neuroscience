# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:39:05 2021
@author: Tobias
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

print("#################  a  #####################")
#a)
img = cv2.imread('brain-noisy.png', 0)
print(type(img))
print(img)
plt.imshow(img, cmap='gray')
plt.show()

median = cv2.medianBlur(img, 5)
plt.imshow(median, cmap='gray')
plt.show()

mask = median > 0
non_zero_pixels = median[mask]

print("#################  b  #####################")
#b)
hist = np.bincount(non_zero_pixels)
log_hist = np.log(hist)
ind = np.arange(np.shape(hist)[0])
plt.plot(ind, hist)
plt.show()
plt.plot(ind, log_hist)
plt.show()

most_freq = np.argmax(hist)
print(type(np.where(hist > np.max(hist) * 0.5)))
print(np.where(hist > np.max(hist) * 0.5))
frequent = np.where(hist > np.max(hist) * 0.5)[0]
max_vis = img[:]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if not np.any(frequent == max_vis[i,j]):
            max_vis[i,j] = 0
plt.imshow(max_vis, cmap='gray')
plt.show()

print("#################  c  #####################")
#c)
k = 3
def init_kmeans(x, k):
    x_ = x.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k).fit(x_)
    mu = kmeans.cluster_centers_
    print(mu)
    mu_ = mu[:, 0]
    nn = kmeans.predict(x_)
    z = np.bincount(nn)
    pi = z / np.sum(z)
    sigma = np.empty(k)
    for i in range(k):
        sigma[i] = np.sqrt(np.mean((x[nn == i] - mu_[i])**2))
    return mu, sigma, pi

def init_constant(x, k):
    mu = np.array([20, 120, 200])
    sigma = np.empty(k)
    kmeans = KMeans(n_clusters=k)
    kmeans.cluster_centers_ = mu.reshape(-1, 1)
    nn = kmeans.predict(x.reshape(-1, 1))
    z = np.bincount(nn)
    pi = z / np.sum(z)
    for i in range(k):
        sigma[i] = np.sqrt(np.mean((x[nn == i] - mu[i])**2))
    return mu, sigma, pi
    
mu, sigma, pi = init_kmeans(non_zero_pixels, k)

print("#################  d  #####################")

def estep(x, mu, sigma, pi):
    rho = np.empty((x.shape[0], mu.shape[0]))
    for j in range(rho.shape[1]):
        rho[:, j] = pi[j] * scipy.stats.norm(mu[j], sigma[j]).pdf(x)
    normalization = np.sum(rho, axis=1)
    norm = np.empty_like(rho)
    for i in range(k):
        norm[:, i] = normalization[:]
    rho = rho / norm
    return rho


#d)
rho = estep(non_zero_pixels, mu, sigma, pi)

def visualize_responsibilities(shape, mask, rho):
    vis = np.zeros(shape)
    vis[mask] = rho
    vis = to_image(vis)
    plt.imshow(vis)
    plt.show()

shape = (img.shape[0], img.shape[1], 3)
visualize_responsibilities(shape, mask, rho)

print("#################  E  #####################")

#e)
def mstep(x, rho):
    K = rho.shape[1]
    N = np.sum(rho, axis=0)
    print(N)
    mu = np.empty(K)
    sigma = np.empty(K)
    pi = np.empty(K)
    for k in range(sigma.shape[0]):
        mu[k] = np.sum(rho[:, k] * x) / N[k]
        sigma[k] = np.sqrt(np.sum(rho[:, k] * (x - mu[k])**2) / N[k])
        pi[k] = N[k] / x.shape[0]
    #print("M-Step:", mu, sigma, pi)
    return mu, sigma, pi

print("#################  f  #####################")

#f)
eps = 0.001
counter = 0
while True:
    print(counter)
    print('Hello')
    print(np.min(rho[:, 0]), np.mean(rho[:, 0]), np.max(rho[:, 0]))
    print(np.min(rho[:, 1]), np.mean(rho[:, 1]), np.max(rho[:, 1]))
    print(np.min(rho[:, 2]), np.mean(rho[:, 2]), np.max(rho[:, 2]))
    mu, sigma, pi = mstep(non_zero_pixels, rho)
    rho_ = estep(non_zero_pixels, mu, sigma, pi)
    counter += 1
    diff = np.mean((rho_ - rho)**2)
    rho = rho_
    if diff < eps:
        break
    
visualize_responsibilities(shape, mask, rho)
print('\nfinal parameters:')
print(mu, sigma, pi)
