import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/mycode/wines.csv')

y = dataset['Class']
y_catog = pd.get_dummies(y)
x=dataset[['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280-OD315_of_diluted_wines','Proline']]

X_train, X_test, y_train, y_test = train_test_split(x,y_catog,test_size=0.15, random_state=42)

model = Sequential()
model.add(Dense(units=64, input_dim=13 , activation="relu"))
model.add(Dense(units=3,  activation="softmax"))
model.compile(optimizer=Adam() , loss='categorical_crossentropy' , metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test) , epochs=100 , verbose=0)

accuracy = model.evaluate(X_train, y_train, verbose=0)
accuracy = accuracy[1]*100

print(accuracy)

import os
os.system(" touch /mycode/accuracy.txt")
os.system("echo {} > /mycode/accuracy.txt".format(accuracy))

model.save('/mycode/modeltrained.pk1')

