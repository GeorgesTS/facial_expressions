import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

data=[]

filepath = "/home/orion/Downloads/facial-landmarks/pred.csv"




def parsef(file):
	data = []
	assert os.path.isfile(file)

	data = np.loadtxt(file, delimiter=',')
	x = data[:, ::2]
	y = data[:, 1::2]
	

	return x,y


X_points,Y_points=parsef(filepath)

plt.plot(X_points[0],-(Y_points[0]),'o')
plt.show()
