# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:34:59 2021

@author: kcarl
"""
import pandas as pd

import keras 
from keras.models import Sequential
from keras.layers import Dense

#Importacion dataset
df= pd.read_csv('datiños.csv')

#Cracion varible dependiente e independiente

x=df.iloc[:,:1].values

y=df.iloc[:,1:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.2, random_state=0)



#Inicializar primer red neuronal artificial
classifier = Sequential()

#Añadir capas de entrada y capa oculta
classifier.add(Dense(units=3,input_dim=1))

#Segunda capa oculta
classifier.add(Dense(units=3))

#Capa de salida
classifier.add(Dense(units=1))

#Compilar la RNA
classifier.compile(optimizer="adam", loss="mean_squared_error")

#Ajuste RNA
classifier.fit(x_train,y_train, batch_size= 6, epochs=20)

y_pred = classifier.predict(x_test)
y_pred1 = classifier.predict([100.0])
y_pred2 = classifier.predict([60.0])