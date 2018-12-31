# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:28:42 2018

@author: rotc_
"""
import numpy as np
from PIL import Image   
from pathlib import Path
from random import shuffle
import glob
from scipy import stats
import shutil
import os

img_train_path = "./Images/train/*.png"

arquivos_train = glob.glob(img_train_path)

labels = np.genfromtxt('train.csv', delimiter=',',skip_header =1 )[:,1]

train_addrs = arquivos_train[0:int(0.9*len(arquivos_train))]
dev_test_addrs = arquivos_train[int(0.9*len(arquivos_train)):]
dev_addrs = dev_test_addrs[0:int(0.5*len(dev_test_addrs))]
test_addrs = dev_test_addrs[int(0.5*len(dev_test_addrs)):]

train_labels = labels[0:int(0.9*len(labels))]
dev_test_labels = labels[int(0.9*len(labels)):]
dev_labels = dev_test_labels[0:int(0.5*len(dev_test_labels))]
test_labels = dev_test_labels[int(0.5*len(dev_test_labels)):]

for i in range(len(train_addrs)):
    folder = 'data/train/{0}'.format(str(int(train_labels[i])))
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copy(train_addrs[i], folder)

for i in range(len(dev_addrs)):
    folder = 'data/validation/{0}'.format(str(int(dev_labels[i])))
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copy(dev_addrs[i], folder)

for i in range(len(test_addrs)):
    folder = 'data/test/{0}'.format(str(int(test_labels[i])))
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copy(test_addrs[i], folder)
 
    