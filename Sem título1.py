# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:11:08 2018

@author: CAST
"""

for addr in arquivos_train:
    for name in labels:
        if '\\'+name in addr:
            print(addr + "->" + labels[name])