

#Function used to transform the CSV TO HDF5



import pandas as pd 
import numpy as np 
import h5py as h5



X_train=pd.read_csv('/home/ozaki/Downloads/facial-landmarks/speaker.csv')
X_test=pd.read_csv('/home/ozaki/Downloads/facial-landmarks/speaker_test.csv')
Y_train=pd.read_csv('/home/ozaki/Downloads/facial-landmarks/listener.csv')
Y_test=pd.read_csv('/home/ozaki/Downloads/facial-landmarks/listener_test.csv')
X_val=pd.read_csv('/home/ozaki/Downloads/facial-landmarks/validation_speaker.csv')
Y_val=pd.read_csv('/home/ozaki/Downloads/facial-landmarks/validation_listener.csv')



#Importing all the different files into one .h5 file,we can select the different datasets by using the appropriate keys.

with h5.File("/home/ozaki/Downloads/facial-landmarks/speaker.h5",'w') as h5:
     h5.create_dataset('speaker_train', data = X_train.values, compression = 'gzip')
     h5.create_dataset('speaker_test', data = X_test.values, compression = 'gzip')
     h5.create_dataset('listener_train', data = Y_train.values, compression = 'gzip')
     h5.create_dataset('listener_test', data = Y_test.values, compression = 'gzip')
     h5.create_dataset('speaker_validation', data = X_val.values, compression = 'gzip')
     h5.create_dataset('listener_validation', data = Y_val.values, compression = 'gzip')




