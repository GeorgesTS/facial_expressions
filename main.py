# USAGE
#python main.py --shape-predictor shape_predictor_68_face_landmarks.dat 

# import the necessary packages

from dir import directory
from detection import detection
import os,fnmatch
import numpy as np

p='/home/orion/Downloads/facial-landmarks'
files=os.listdir(p)
ext="*.[Mm][Pp]4" 
path=[]






path, csv_names=directory(p,files)
counter=np.size(path)
detection(counter,path,csv_names)

