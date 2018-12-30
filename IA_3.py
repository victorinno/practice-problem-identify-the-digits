# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:58:59 2018

@author: rotc_
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import h5py
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist


np.random.seed(1)
batch_size = 128
num_classes = 10
epochs = 1

f = h5py.File('data.hdf5', 'r')

(X_train ,Y_train), (X_test,Y_test) = mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#X_train = np.array(X_train) / 255.0
#X_test = np.array(X_test) / 255.0

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=epochs, callbacks = [cp_callback], batch_size=batch_size, shuffle="batch")

test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

