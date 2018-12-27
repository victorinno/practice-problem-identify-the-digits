# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:25:17 2018

generation of a data set in a h5 file.

@author: Floriano Victor Larangeira Peixoto
"""

import h5py
import numpy as np
from PIL import Image   
from pathlib import Path
from random import shuffle
import glob

shuffle_data = True

hdf5_path = "data.hdf5"

img_train_path = "./Images/train/*.png"
img_test_path = "./Images/test/*.png"

arquivos_train = glob.glob(img_train_path)
arquivos_test = glob.glob(img_test_path)


labels = np.genfromtxt('train.csv', delimiter=',',skip_header =1 )[:,1]


train_addrs = arquivos_train[0:int(0.8*len(arquivos_train))]
dev_test_addrs = arquivos_train[int(0.8*len(arquivos_train)):]
dev_addrs = dev_test_addrs[0:int(0.5*len(dev_test_addrs))]
test_addrs = dev_test_addrs[int(0.5*len(dev_test_addrs)):]

train_labels = labels[0:int(0.8*len(labels))]
dev_test_labels = labels[int(0.8*len(labels)):]
dev_labels = dev_test_labels[0:int(0.5*len(dev_test_labels))]
test_labels = dev_test_labels[int(0.5*len(dev_test_labels)):]


train_shape = (len(train_addrs), 28,28,3)
dev_shape = (len(dev_addrs), 28,28, 3)
test_shape = (len(test_addrs), 28,28, 3)

f = h5py.File(hdf5_path, mode='w')

f.create_dataset("train_img", train_shape, np.uint8)
f.create_dataset("dev_img", dev_shape, np.uint8)
f.create_dataset("test_img", test_shape, np.uint8)

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key. 
f.create_dataset("train_labels", (len(train_addrs),), np.uint8)
f["train_labels"][...] = train_labels

f.create_dataset("dev_labels", (len(dev_addrs),), np.uint8)
f["dev_labels"][...] = dev_labels

f.create_dataset("test_labels", (len(test_addrs),), np.uint8)
f["test_labels"][...] = test_labels


for i in range(len(train_addrs)):
  
    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)) )
    
    addr = train_addrs[i]
    image = Image.open(addr,'r')
    image = image.convert('RGB')
    train = np.array(image)
    #flat_data = pixel_values.reshape(pixel_values.shape[0], -1).T
    #train = flat_data / 255.
    f["train_img"][i, ...] = train 

for i in range(len(dev_addrs)):
  
    if i % 1000 == 0 and i > 1:
        print ('Dev data: {}/{}'.format(i, len(dev_addrs)) )
    
    addr = train_addrs[i]
    image = Image.open(addr,'r')
    image = image.convert('RGB')
    train = np.array(image)
    #flat_data = pixel_values.reshape(pixel_values.shape[0], -1).T
    #train = flat_data / 255.
    f["dev_img"][i, ...] = train 

for i in range(len(test_addrs)):
  
    if i % 1000 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)) )
    
    addr = train_addrs[i]
    image = Image.open(addr,'r')
    image = image.convert('RGB')
    train = np.array(image)
    #flat_data = pixel_values.reshape(pixel_values.shape[0], -1).T
    #train = flat_data / 255.
    f["test_img"][i, ...] = train 
    
    
f.close()
