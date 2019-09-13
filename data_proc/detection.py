from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import pandas as pd 
import os,fnmatch


p='/home/ozaki/Downloads/facial-landmarks'
files=os.listdir(p)
ext="*.avi" 
path=[]

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


def detection(counter,path,csv_names):
    final = np.array([])
    c=0
    for i in range (0,counter):
        
        c=c+1
        print(path[i])
        cap = cv2.VideoCapture(path[i])

        while True:
            _, image = cap.read()
                        
            if image is None:
                break
                print("No image available")



            else:

                            
                image = imutils.resize(image, width=500)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale image
                rects = detector(gray, 1)
                            
                print(rects)

                    # loop over the face detections
                for (i, rect) in enumerate(rects):

                        # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                    shape = predictor(gray, rect)
                                
                    shape = face_utils.shape_to_np(shape)
                            
                    
                    print("coordinates of face")
                            
                    shape = np.reshape(shape, (1, shape.size))
                    #print(shape.shape)
                    if final.size == 0:
                        final=shape

                    else:
                        
                        final=np.concatenate((final,shape),axis=0)

        print(final.shape)
        pd.DataFrame(final).to_csv(os.path.join(p,csv_names[c-1]))
        print("Points writen to file", csv_names[c-1] )   
        final=np.array([])
