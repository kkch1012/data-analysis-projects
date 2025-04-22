# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:21:49 2025

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras

x = np.array([[0,0],[0,1],[1,0],[0,2],[1,1],[2,0]])
y = np.array([  0,  0,  0,  1,  1,  1])

model = Sequential()
model.add(Dense(1,input_dim=2,activation='sigmoid'))


model.compile(optimizer='sgd',loss='binary_crossentropy',
              metrics=['binary_accuracy'])

model.fit(x,y,epochs=2000)






