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


np.random.seed(1)
batch_size = 128
num_classes = 10
epochs = 12

f = h5py.File('data.hdf5', 'r')

X_train = f['train_img']
Y_train = f['train_labels']
X_test = f['dev_img']
Y_test = f['dev_labels']

X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,3)),
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

model.fit(X_train, Y_train, epochs=epochs, callbacks = [cp_callback], batch_size=batch_size)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
