# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 22:44:15 2018

@author: rotc_
"""

import os
import re

img_train_path = "./Images/train/*.png"

arquivos_train = glob.glob(img_train_path)

reg = '(\d{1,10})'

for a in arquivos_train:
    m = re.search(reg, a)
    novoNome = './Images/train/{0}.png'.format(m.group(0).zfill(6))
    os.rename(a, novoNome)