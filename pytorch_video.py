import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils,models
import torch.nn.functional as F
from pytorch_model import *
from torch.optim import lr_scheduler
from torchvision import transforms, utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_dataset import *
from tqdm.notebook import tqdm
from tqdm.auto import tqdm as tq
import time
import numpy as np
import os
import cv2
from pytorch_model import *
from Inputs import *
import time

##################################################################3


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
device = torch.device("cuda")
path = 'modelf_epoch_99.pth'
fine_mod = FineAP(3).to(device)





def get_vid(vid,n_frames=16):
   
    cap = cv2.VideoCapture(vid)
    opframes=[]
    inp = []
    frames = []
    vid_frames = []
    i=0
       
    while True:
        ret, frame = cap.read()
    
        
        if not ret:
            break
        opframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frames.append(cv2.cvtColor(cv2.resize(frame,(64,64)), cv2.COLOR_BGR2RGB))
               
                
               

        #Omitting last frame because gaze coordinates of the last frame are not available 
                
    frx = list(range(0,len(frames)-1,n_frames))
       
    for i in range(len(frx)):
        try:
            inp.append( np.array( frames[frx[i]:frx[i+1]] )) 
        except IndexError:
            pass
        
        
    vid_frames.append(opframes)
    cap.release()
    cv2.destroyAllWindows()
        
        
        
    return np.array(inp),vid_frames


def pred_coords(predict,test_frames,test_vid=None):
    
    
    predout = predict.squeeze()
    j=0
    predframe = []
    coords_list = get_coords(test_vid)
    for arrframes in predout:
        for frame in arrframes:

            nx = np.argmax((np.max(frame,axis=0)))
            ny = np.argmax((np.max(frame,axis=1)))
    
            x = 320*nx + 160
            y = 180*ny + 90
            try:
                a,b = coords_list[j]
                #print(len(test_frames[i]),len(get_coords(path)) )
                frametemp = cv2.rectangle(test_frames[j],(x-160,y-90),(x+160,y+90),(0, 255, 0), 2)
                predframe.append(cv2.rectangle(frametemp,(a-30,b-30),(a+30,b+30),(255, 0, 0), 2))
                j+=1
            except IndexError:
                pass
            
    return predframe



def get_coords(video,path):
    inparr = []
    gridarr=[]
    inpgrid = []
    path_file = path+video[:-4]+'.txt'
    with open(path_file, "r") as f:
       
        
        content = f.readlines()
        for i in range(len(content)):
            
      #grid = np.zeros((4,4))
            x,y = content[i].strip().split(',')[0:2]
            x,y = int(x),int(y)
            inparr.append([x,y])
    return inparr



def AAError(pred,true):
    #print(pred_coords)
    #print(true_coords)
    z = 1280/np.tan(np.pi/6)
    #print(z)
    #pred = pred_coords.append(z)
    #true = true_coords.append(z)
    #print(pred,true)
    #print(pred,true)
    n= np.cross(pred, true)
    err = np.arctan((np.sqrt(np.sum(n**2)))/(np.dot(pred,true)))
    return err*180/np.pi

def cos(pred,true):
    pred = pred/np.linalg.norm(pred)
    true = true/np.linalg.norm(true)
    
    err = np.arccos(np.dot(pred,true))
    return err*180/np.pi
def coarse_vid(model_path):
    path = 'test'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
    device = torch.device("cuda")
    coarse_mod = CoarseAP(3).to(device)
    coarse_mod.load_state_dict(torch.load(model_path))
    for video in os.listdir(path+'/'+'test_videos'):
        #print(video)
        opframes=[]
        frames = []
        preds=[]
        test_array,test_frames = get_vid(path+'/test_videos/')
        test_array = np.rollaxis(test_array,-1,1)
        test_array = torch.tensor(test_array,dtype=torch.float32,requires_grad=False)
        coarse_mod.eval()
        with torch.no_grad():
            pred = coarse_mod(test_array).squeeze()
            predframe = pred_coords(pred,path,test_frames)
            out = cv2.VideoWriter('test_course_val' + video,cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280,720))
            for i in range(len(predframe)):
                out.write(cv2.cvtColor(predframe[i], cv2.COLOR_BGR2RGB))
            out.release()
            
            
            
def fine_vid(model_path):
    errdf = pd.DataFrame([])
    for i,path in enumerate(['gaze_data/test/','../gaze_data/val/']):
    
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
        device = torch.device("cuda")
        fine_mod = FineAP(3).to(device)
        fine_mod = nn.DataParallel(fine_mod, device_ids=[0,1,2,3,4,5,6,7,8,9])
        fine_mod.load_state_dict(torch.load(model_path))
        dist=0
        distlis=[]
        aae = []
        coserr = []
        totaldist=[]
        totalaae=[]
        
        z = 1280/np.tan(np.pi/6)
        for video in os.listdir(path+'videos/'):
            if video.endswith('.avi'):
            #print(video)
                coords = get_coords(video,path+'labels/')
                opframes=[]
                frames = []
                preds=[]
                test_array,test_frames = get_vid(path+'videos/'+video,n_frames=2)
                test_array = np.rollaxis(test_array,-1,1)
                test_array = torch.tensor(test_array,dtype=torch.float32,requires_grad=False)
            #print(test_array.shape)
                fine_mod.eval()
                with torch.no_grad():
                    for arr in test_array:
                        p = fine_mod(arr[None]).squeeze()
                        preds.append(p.cpu().numpy())

                j=0
                predframe = []
                for arrframes in preds:
                    for frame in arrframes:

                        nx = np.argmax((np.max(frame,axis=1)))
                        ny = np.argmax((np.max(frame,axis=0)))
                        x = int(round(ny*20))
                        y = int(round(nx*11.25))
                        a,b = coords[j]

                        dist=np.sqrt(np.square(x-a)+np.square(y-b))
                        distlis.append(dist)
                        pred,true = [x,y,z],[a,b,z]
                        aae.append(AAError(pred,true))
                        
                        #coserr.append(cos(pred,true))
                        #print('coserr',cos(pred,true),pred,true)

                    #frametemp = cv2.rectangle(test_frames[j],(x-25,y-25),(x+25,y+25),(0, 255, 0), 2)
                    #predframe.append(cv2.rectangle(frametemp,(a-25,b-25),(a+25,b+25),(255, 0, 0), 2))

                        j+=1

            #out = cv2.VideoWriter(f'{video}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280,720))
            #for i in range(len(predframe)):
            #    out.write(cv2.cvtColor(predframe[i], cv2.COLOR_BGR2RGB))

            #out.release()
                print(np.mean(distlis))
                print(video,'AvgAE =' ,np.mean(aae))
                #print('cos=',np.mean(coserr))
                totaldist.append(np.mean(distlis))
                totalaae.append(np.mean(aae))
            
        print(i)
        print('length',len(totaldist))
        errdf[str(i)+'Video'] = pd.Series([vid for vid in os.listdir(path+'videos/') if vid.endswith('.avi')])
        errdf[str(i)+'Dist'] = pd.Series(totaldist)
        errdf[str(i)+'AAE'] = pd.Series(totalaae)
    errdf.to_csv('errors_(21-8).csv')        
           
    



################################################################################

fine_vid('models/pytorch_models/modelf_(21-8).pth')

