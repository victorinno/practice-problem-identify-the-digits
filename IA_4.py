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

#https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

K.set_image_dim_ordering('th')


np.random.seed(1)

batch_size = 32
num_classes = 10
epochs = 5

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=r"./data/train/",
    target_size=(28, 28),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    directory=r"./data/validation",
    target_size=(28,28),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=r"./data/test/",
    target_size=(28,28),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

#result_generator = test_datagen.flow_from_directory(
#    directory=r"./data/final/test/",
#    target_size=(28,28),
#    color_mode="rgb",
#    batch_size=1,
#    class_mode=None,
#    shuffle=False,
#    seed=42
#)

def softMaxAxis1(x):
    return keras.activations.softmax(x,axis=1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 28, 28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(keras.layers.Dense(output_dim=10, activation=softMaxAxis1))

# the model so far outputs 3D feature maps (height, width, features)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.n//32,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.n//32,
                    epochs=1
)

valid_generator.reset()
model.evaluate_generator(generator=valid_generator, steps=len(valid_generator))

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

#result_generator.reset()
#pred_result=model.predict_generator(result_generator,verbose=1)
#
#predicted_class_indices=np.argmax(pred_result,axis=1)
#
#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]
#
#filenames=test_generator.filenames
#results=pd.DataFrame({"Filename":filenames,
#                      "Predictions":predictions})
#results.to_csv("results.csv",index=False)
