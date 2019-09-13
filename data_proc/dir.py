import os,fnmatch
import numpy as np
p='/home/ozaki/Downloads/facial-landmarks'
files=os.listdir(p)
ext="*.avi" 
path=[]


def directory(p,files):

        path=[]
        csv_names=[]
        

        for entry in files:
                if fnmatch.fnmatch(entry, ext):

                
                        path.append(os.path.join(p,entry))
                        name=entry.replace('.avi', '.csv')
                        
                
                        csv_names.append(os.path.join(p,name))
        
        return path,csv_names
        

