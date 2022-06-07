# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:40:16 2021

@author: Ranak Roy Chowdhury
"""

import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



dimensions = 2
data = []
for i in range(1, dimensions + 1):
    filename = 'AtrialFibrillationDimension' + str(i) + '_TEST.arff' # change to '_TRAIN.arff' to build training data
    file = open(filename, "r")
    dataset = arff.load(file)
    dataset = np.array(dataset['data'])
    data.append(dataset[ : , 0 : -1])
data = np.array(data)
data = np.transpose(data, (1, 2, 0))
print(data.shape)
np.save('X_test.npy', data)



label = np.array(dataset[ : , -1])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded.shape)
np.save('y_test.npy', onehot_encoded)