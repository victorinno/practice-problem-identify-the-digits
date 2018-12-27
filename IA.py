# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:35:11 2018

@author: Floriano Victor Larangeira Peixoto
"""

import math
import numpy as np
import h5py

import tensorflow as tf
from tensorflow.python.framework import ops

np.random.seed(1)

f = h5py.File('data.hdf5', 'r')

X_train = f['train_img']
Y_train = f['train_labels']
X_test = f['dev_img']
Y_test = f['dev_labels']
