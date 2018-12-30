# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:52:42 2018

@author: rotc_
"""
import h5py

import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras

K.set_image_dim_ordering('th')


np.random.seed(1)

batch_size = 32
num_classes = 10
epochs = 5

f = h5py.File('data.hdf5', 'r')
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = np.array( f['train_img'])
Y_train = np.array( f['train_labels'])
#X_dev = np.array( f['dev_img'])
#Y_dev = np.array( f['dev_labels'])
X_test = np.array( f['test_img'])
Y_test = np.array( f['test_labels'])

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True, rotation_range=90)
# fit parameters from data
#datagen.fit(X_train)

trainer = datagen.flow(X_train, Y_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), activation='relu'))

#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        trainer,
        steps_per_epoch=2000 // batch_size,
        epochs=50)
        #validation_data=validation_generator,
        #validation_steps=800 // batch_size)
## configure batch size and retrieve one batch of images
#for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=20000):
#	# create a grid of 3x3 images
#	for i in range(0, 9):
#		pyplot.subplot(330 + 1 + i)
#		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
#	# show the plot
#	pyplot.show()
#	break

train_loss, train_acc = model.evaluate(X_train, Y_train)
test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test loss:', test_loss)
print('Train loss:', train_loss)
print('Test accuracy:', test_acc)
print('Train accuracy:', train_acc)

f.close()