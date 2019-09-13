import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image


filepath = "/home/ozaki/Downloads/facial-landmarks/fin.csv"




def parsef(file):
	data = []
	assert os.path.isfile(file)

	data = np.loadtxt(file, delimiter=',')
	x = data[:, ::2]
	y = data[:, 1::2]
	

	return x,y


X_points, Y_points=parsef(filepath)


path="/home/ozaki/Downloads/facial-landmarks/plots/"

for i in range (0,len(X_points)):
	plt.plot(X_points[i],-(Y_points[i]),'o')
	plt.savefig(path+str(i)+'.png')
	plt.clf()
	
