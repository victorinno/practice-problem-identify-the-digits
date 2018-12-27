# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:58:59 2018

@author: rotc_
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import h5py

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import losses
#import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

np.random.seed(1)

f = h5py.File('data.hdf5', 'r')

X_train = f['train_img']
Y_train = f['train_labels']
X_test = f['dev_img']
Y_test = f['dev_labels']

X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=1500, callbacks = [cp_callback], batch_size=512)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
