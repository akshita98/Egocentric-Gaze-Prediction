import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, Conv3D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten,MaxPooling3D,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10

from keras.utils import plot_model
from keras.optimizers import SGD
import numpy as np
import os
from keras.losses import BinaryCrossentropy

from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from gaze_model import *
from Inputs import *
import os
##################################################################3

model_path = 'fine_preds(200epch).h5'
#coarse_mod = coarse_model()
fine_mod = fine_model()
fine_mod.load_weights(model_path)



def pred_coords(predict,test_frames,test_vid=None):
    
    
    predout = predict.squeeze()
    j=0
    predframe = []
    coords_list = get_coords(test_vid)
    for arrframes in predout:
        for frame in arrframes:

            nx = np.argmax((np.max(frame,axis=1)))
            ny = np.argmax((np.max(frame,axis=0)))
    
            x = 320*ny + 160
            y = 180*nx + 90
            try:
                a,b = coords_list[j]
                #print(len(test_frames[i]),len(get_coords(path)) )
                frametemp = cv2.rectangle(test_frames[j],(x-160,y-90),(x+160,y+90),(0, 255, 0), 2)
                predframe.append(cv2.rectangle(frametemp,(a-30,b-30),(a+30,b+30),(255, 0, 0), 2))
                j+=1
            except IndexError:
                pass
            
    return predframe



def get_coords(video):
    inparr = []
    gridarr=[]
    inpgrid = []
    path_file = path+'/test_labels/'+video[:-4]+'.txt'
    with open(path_file, "r") as f:
       
        
        content = f.readlines()
        for i in range(len(content)):
            
      #grid = np.zeros((4,4))
            x,y = content[i].strip().split(',')[0:2]
            x,y = int(x),int(y)
            inparr.append([x,y])
    return inparr

################################################################################

path = 'test'
for video in os.listdir(path+'/'+'test_videos'):
    print(video)
    opframes=[]
    frames = []
    preds=[]
   
    test_array,test_frames = get_frames(vidpath=path+'/test_videos/'+video,train=False)
    #np.save('fine_test',test_array)
    #np.save('fine_frames',test_frames)
    for arr in test_array:
        p = fine_mod.predict(arr[None])
        preds.append(p)
    #preds = coarse_mod.predict(test_array)
    #predframe = pred_coords(pred,test_frames,test_vid = video)
    #np.save('coarse_preds(10epch)',preds)
    #out = cv2.VideoWriter('test_course_val' + video,cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280,720))

    #print(predframe[0].shape)
    #for i in range(len(predframe)):
       
     #   out.write(cv2.cvtColor(predframe[i], cv2.COLOR_BGR2RGB))
        
    
    #out.release()
    print(test_array.shape,'done')
    print(len(test_frames),'done')
    print(video)
    

   
