#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense ,BatchNormalization,Dropout
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam


dataset = pd.read_csv('winequalityN.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1:].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



X_test = X_test / 10
X_test = X_test.reshape(-1,4,3,1)

X_train = X_train / 10
X_train = X_train.reshape(-1,4,3,1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()








#Building the CNN
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(16, 2, 2, input_shape = (4,3, 1), activation = 'relu'))
classifier.add(Convolution2D(16, 2, 2, activation = 'relu'))

# Step 2 - Pooling
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(BatchNormalization())
#classifier.add(Dropout(0.4))


# Adding a second convolutional layer
classifier.add(Convolution2D(16, 1, 1, activation = 'relu'))
classifier.add(Convolution2D(16, 1, 1, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(BatchNormalization())
#classifier.add(Dropout(0.4))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 16, activation = 'relu'))
#classifier.add(Dropout(0.4))


classifier.add(Dense(output_dim = 8, activation = 'softmax'))
#classifier.add(Dropout(0.4))


# Compiling the CNN

classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Image Data generator
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image 
        width_shift_range=0, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_test)

#OneHot Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()


classifier.fit_generator(datagen.flow(X_train, y_train , batch_size=86),
                         steps_per_epoch= 30,validation_data = (X_test,y_test),
                         epochs=30)


# # ??????????????????????????????

# In[ ]:




