#!/usr/bin/env python3					###### Should I put this? (to run the program as an executable or sth.)

import numpy as np
import scipy as scp
from matplotlib import pyplot as plt
import skimage as skm  					# image manipulating routines
import os 								# OS interaction routines

test_pic = skm.data.camera()
print(test_pic)

# set matplotlib as GUI plugin
# I have actually very little understanding of how io.imshow actually works...
#skm.io.use_plugin('matplotlib')
# e.g. even without this specification the program chooses matplot.

# Using distribution independet path names
filename = os.path.join(skm.data_dir, 'data/brain.png')
print(filename)
#brain = skm.io.imread(filename)		##### -> why does this not work?


brain = skm.io.imread('./data/brain.png')
brain_binary = brain >= 100
skm.io.imshow(brain_binary, cmap='Greys_r')
skm.io.show()							#####  -> to do: plot proer as subplots

gradient = np.linspace(np.zeros(100), np.ones(100), num=255, axis=1)
skm.io.imshow(gradient, cmap='Greys_r')
skm.io.show()							#####  -> to do: plot proer as subplots
