# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:35:31 2019

@author: HP
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('C:\\Users\\HP\\Desktop\\u_datasets\\Artificial_Neural_Networks\\Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
onehotencoder= OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
#adding the input layer and the 1st hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#adding the 2ns hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
##adding output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)


#prediting the test result
y_pred=classifier.predict(x_test)
y_pred
y_pred=(y_pred > 0.5)

#confusion matrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





