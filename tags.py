# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 16:23:27 2018

@author: pranay
"""

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import _pickle
import requests
import concurrent.futures
import urllib.request


#Create tags dictionary with id as key and relevant tags as values
tags_dict = {}
with open('All_Tags.txt','r', encoding='utf-8') as f1:
    for line in f1:
        print(line)
        line = line.split()
        tags_dict[line[0]] = line[1:]
        
with open('tags_dict.pickle','wb') as pfile:
    _pickle.dump(tags_dict, pfile)

with open('tags_dict.pickle','rb') as pfile:
    tags_dict = _pickle.load(pfile) 

        
'''data = []
labels = []
model = VGG16(include_top=True,weights='imagenet')
model.layers.pop()
model.outputs = [model.layers[-1].output]'''
#i = 0
urls_dict = {}
with open('NUS-WIDE-urls.txt','r') as f2:
    next(f2)
    for line in f2:
        #i=i+1
        line_list = line.split()
        if line_list[1] in tags_dict.keys():
            #print(line_list[3])
            if line_list[4]!='null':
                urls_dict[line_list[1]] = line_list[4]
                '''
                im = requests.get(line_list[3], allow_redirects=False)
                if im.status_code==200:
                    img_data=im.content
                    with open('images/'+line_list[1]+'.jpg', 'wb') as handler:
                            handler.write(img_data)
                    img_path = 'images/'+line_list[1]+'.jpg'
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_data = image.img_to_array(img)
                    img_data = np.expand_dims(img_data, axis=0)
                    img_data = preprocess_input(img_data)
                    
                    vgg16_feature = model.predict(img_data)
                    data.append(vgg16_feature)
                    labels.append(tags_dict[line_list[1]])
                    print('Done: {}/{}'.format(i,269648))'''
                    
with open('urls_dict.pickle','wb') as dfile:
    _pickle.dump(urls_dict, dfile)
    
with open('urls_dict.pickle','rb') as dfile:
    urls_dict = _pickle.load(dfile) 
    
def getimg(count):
    localpath = 'images/{0}.jpg'.format(list(urls_dict.keys())[count])
    if requests.get(list(urls_dict.values())[count], allow_redirects=False).status_code==200:
        urllib.request.urlretrieve(list(urls_dict.values())[count], localpath)
        '''img = image.load_img(localpath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        vgg16_feature = model.predict(img_data)
        data.append(vgg16_feature)
        labels.append(tags_dict[list(urls_dict.keys())[count]])'''
        print('Done')
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as e:
    for i in range(257784):
        e.submit(getimg, i)

    

                
        
        
        
        

