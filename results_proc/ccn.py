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



batch_size = 128
#num_classes = 10
epochs = 2000

 


with h5py.File('/home/ozaki/Downloads/facial-landmarks/speaker.h5','r') as hdf:
    ls=list(hdf.keys())
    data1=hdf.get('speaker_train')
    x_train=np.array(data1)
    data2=hdf.get('speaker_test')
    x_test=np.array(data2)
    data3=hdf.get('listener_train')
    y_train=np.array(data3)
    data4=hdf.get('listener_test')
    y_test=np.array(data4)
    data5=hdf.get('speaker_validation')
    x_val=np.array(data5)
    data6=hdf.get('listener_validation')
    y_val=np.array(data6)
    



print(x_train.shape)
print(x_test.shape)
print(x_val.shape)



x_train=x_train.reshape(269999,1,136,1)
x_test=x_test.reshape(2999,1,136,1)



#the y are for the output layer this is why they have 2 dimensions
y_train = y_train.reshape(269999,136)
y_test = y_test.reshape(2999,136)



print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



model = Sequential()
model.add(Conv2D(32, kernel_size=(1,1),
                 activation='relu',
                 input_shape=(1,136,1)))
model.add(Conv2D(64, (1,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Dropout(0.25))
model.add(Flatten())
#the dense layer corresponds to the ouput parameters (y_train,y_test)
model.add(Dense(164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(136))
######CHECK DENSE DIMENSIONS$$$####
model.summary()
print(x_train.shape)








model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])



history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# Saving Model
model.save('cnn_final.h5')
