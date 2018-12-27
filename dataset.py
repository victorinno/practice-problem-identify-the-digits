# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import h5py
import numpy as np
from PIL import Image   
from pathlib import Path

data = None

p = Path('./Images/train').glob("*.png")

image = Image.open('./Images/train\\1.png','r')
image = image.convert('RGB')

pixel_values = np.array(image.getdata())

print(pixel_values.shape)

flat_data = pixel_values.reshape(pixel_values.shape[0], -1).T


train = flat_data / 255.

print(train.shape)   