# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 23:01:02 2018

@author: pranay
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(include_top=True,weights='imagenet')
model.layers.pop()

model.outputs = [model.layers[-1].output]
#model.layers[-1].outbound_nodes = []

img_path = 'E:/ML projects/Instagram/Screenshot_20180519-211720.png'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_feature2 = model.predict(img_data)


print(vgg16_feature.shape)