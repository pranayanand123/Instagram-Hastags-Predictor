# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:40:47 2018

@author: pranay
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

class NN:
	def build_model(width, height, depth, classes, finalAct="softmax"):
            model = Sequential()
            inputShape = (height, width, depth)
            chanDim = -1
            
            model.add(Conv2D(32, (3, 3), padding="same",
                    input_shape=inputShape))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(0.25))
              
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
     
            
            model.add(Conv2D(128, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(128, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            model.add(Flatten())
            model.add(Dense(1024))
            model.add(Activation("relu"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
     
            # use a *softmax* activation for single-label classification
            # and *sigmoid* activation for multi-label classification
            model.add(Dense(classes))
            model.add(Activation(finalAct))
     
            
            return model
