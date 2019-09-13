

from keras.applications.vgg16 import VGG16
from keras import models
from keras import optimizers
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
import os
from PIL import Image
import cv2
from dataset import im,data

batch_size=30
epochs=30


filename1='/home/orion/Downloads/facial-landmarks/X_TRAIN/'
filename2='/home/orion/Downloads/facial-landmarks/Y_TRAIN/'
filename3='/home/orion/Downloads/facial-landmarks/X_TEST/'
filename4='/home/orion/Downloads/facial-landmarks/Y_TEST/'
files1=os.listdir(filename1)
files2=os.listdir(filename2)
files3=os.listdir(filename3)
files4=os.listdir(filename4)


X_train=data(filename1,files1)
Y_train=data(filename2,files2)
X_test=data(filename3,files3)
Y_test=data(filename4,files4)


data1=im(X_train)
data2=im(Y_train)
data3=im(X_test)
data4=im(Y_test)


x_train=np.asarray(data1)
y_train=np.asarray(data2)
x_test=np.asarray(data3)
y_test=np.asarray(data4)




#Load the VGG model
vgg_conv=VGG16(weights='imagenet',include_top=False,input_shape=(480,640,3))

for layer in vgg_conv.layers[:-3]:
    layer.trainable=False


# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


#Creation of the model

model=models.Sequential()


#Add the VGG

model.add(vgg_conv)

#Add custom layers

model.add(Conv2D(32, kernel_size=(1,1),
                 activation='relu'))
model.add(Conv2D(64, (1,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))
#model.add(Flatten())
#the dense layer corresponds to the ouput parameters (y_train,y_test)
model.add(Dense(164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(136))

#------------------------------SUMMARY--------------------#
model.summary()



model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
    

          






# Save the model
model.save('this_is_images.h5')


