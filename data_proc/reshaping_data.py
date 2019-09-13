import os
from PIL import Image
import numpy as np
from copy import copy, deepcopy


filepath1 = "/home/ozaki/Downloads/facial-landmarks/speaker.csv"
filepath2 = "/home/ozaki/Downloads/facial-landmarks/speaker_test.csv"
filepath3 = "/home/ozaki/Downloads/facial-landmarks/validation_speaker.csv"
filepath4 = "/home/ozaki/Downloads/facial-landmarks/listener.csv"
filepath5 = "/home/ozaki/Downloads/facial-landmarks/listener_test.csv"
filepath6 = "/home/ozaki/Downloads/facial-landmarks/validation_listener.csv"

size_out=[256, 256]

def parsef(file):
	data = []
	assert os.path.isfile(file)

	data = np.loadtxt(file, delimiter=',')
	x = data[:, ::2]
	y = data[:, 1::2]
	z = np.ones((x.shape[0], x.shape[1]))
	# z = (x+y)/2
	data = np.dstack((x, y, z))

	return data



def parsef_withoutz(file):
	data = []
	assert os.path.isfile(file)

	data = np.loadtxt(file, delimiter=',')
	# data = np.genfromtxt(path_to_mocap_file, delimiter=' ', usecols=range(54), invalid_raise=False, dtype=float)
	x = data[25::50, ::2]
	y = data[25::50, 1::2]
	
	data = np.dstack((x, y))

	return data





def change_dims():
    x_train=parsef(filepath1)
    x_test=parsef(filepath2)
    x_val=parsef(filepath3)
    
    
    return x_train,x_test,x_val

change_dims()



	



def change_dims_witoutz():

	y_train=parsef_withoutz(filepath4)
	y_test=parsef_withoutz(filepath5)
	y_val=parsef_withoutz(filepath6)

		
	return y_train,y_test,y_val




def lstm_dataset_y(file):
	data = []
	assert os.path.isfile(file)

	data = np.loadtxt(file, delimiter=',')
	dataset = data[::3, :]


	

	return dataset


def lstm_dataset_x(file):
	dataset=[]
	shape=(1200,136)
	
	a = []
	assert os.path.isfile(file)
	data = np.loadtxt(file, delimiter=',')

	k=0
	l=-1
	i=0

	while (i<len(data)):
		k=0
		while(k<3):
			dataset.append(data[i])
			i=i+1
			k=k+1
	
			if( i % 3==0):
				l=l+1

		a.append(dataset)
		dataset=[]




		
		
		
	return a


def lstm_assign_x():
	x_train=lstm_dataset_x(filepath1)
	x_test=lstm_dataset_x(filepath2)
	x_val=lstm_dataset_x(filepath3)

	return x_train,x_test,x_val

def lstm_assign_y():
	y_train=lstm_dataset_y(filepath4)
	y_test=lstm_dataset_y(filepath5)
	y_val=lstm_dataset_y(filepath6)


	return y_train,y_test,y_val
    
    

