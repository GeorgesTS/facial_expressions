import numpy as np 
import h5py

#example code of how to read a .h5 file and transform it into a numpy array in order to use it with a CNN

with h5py.File('/home/orion/Downloads/facial-landmarks/speaker.h5','r') as hdf:


    ls=list(hdf.keys())
    print(ls)
    data=hdf.get('speaker_test')
    la=np.array(data)
    print(data.shape)
    lolo=hdf.get('speaker_train')
    lali=np.array(lolo)
    print(lali.shape)
