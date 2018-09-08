# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 16:23:27 2018

@author: pranay
"""

import requests
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import _pickle
from keras.applications.vgg16 import VGG16


#Create tags dictionary with id as key and relevant tags as values
tags_dict = {}
with open('All_Tags.txt','r', encoding='utf-8') as f1:
    for line in f1:
        print(line)
        line = line.split()
        tags_dict[line[0]] = line[1:]
        
with open('tags_dict.pickle','wb') as pfile:
    _pickle.dump(tags_dict, pfile)
        

        

Y = []
model = VGG16(include_top=True,weights='imagenet')
model.layers.pop()
model.outputs = [model.layers[-1].output]
with open('NUS-WIDE-urls.txt','r') as f2:
    next(f2)
    for line in f2:
        line_list = line.split()
        if line_list[1] in tags_dict.keys():
            #print(line_list[3])
            if line_list[3]!='null':
                im = requests.get(line_list[3], allow_redirects=False)
                if im.status_code==200:
                    img_data=im.content
                    with open(line_list[1]+'.jpg', 'wb') as handler:
                            handler.write(img_data)
                    
                    
                
                
        
        
        
        

