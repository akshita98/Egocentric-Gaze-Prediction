import os
import numpy as np
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd


def create_inp(cap,n_frames,overlapping=False):
    
    vid_vector=[]
    frames = []
    i=0
       
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        frames.append(cv2.cvtColor(cv2.resize(frame,(64,64)), cv2.COLOR_BGR2RGB))

    if overlapping:
        for i in range(len(frames)-n_frames):
            vid_vector.append(np.array(frames[i:i+n_frames]))


        #Omitting last frame because gaze coordinates of the last frame are not available 
    else:
        frx = list(range(0,len(frames)-1,n_frames))
 
        for i in range(len(frx)):
            try:
                vid_vector.append( np.array( frames[frx[i]:frx[i+1]] )) 
                
            except IndexError:
                pass

    return np.array(vid_vector)


def get_frames(vidpath = None,overlapping = False,train=True,n_frames=16):
    frlen=[]
    vlen =  []
    #print(os.listdir(vidpath))
    if train:
        inp = np.zeros(shape=(0,n_frames,64,64,3))
        vid_frames = []

        for vid in os.listdir(vidpath):
            
            if vid.endswith('.avi'):
                #print('vid=',vid)
                cap = cv2.VideoCapture(vidpath+'/'+vid)
                vid_vector = create_inp(cap,n_frames)
               
                inp = np.append(inp,vid_vector,axis=0)
 
    cap.release()
    cv2.destroyAllWindows()
    return inp


def gaussian_k(x0,y0,sigma, width, height):
       
        x = np.arange(0, width, 1, float) ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_hm(height, width ,coords,s=1.5):

    inpgrid = []
    hm = np.zeros((height, width), dtype = np.float32)
    
    if not np.array_equal(coords, [-1,-1]):
         
        hm[:,:] = gaussian_k(coords[0],coords[1],
                                        s,width, height)
    else:
        hm[:,:] = np.zeros((height,width))
        

    return hm


def get_grids(labels_path = None,vidpath=None,n_frames=16):
    frlen=[]
    vlen =  []
    inpgrid = []
    gridarr=[]
    heatmap = []

    gridlist = os.listdir(labels_path)
    vidlist = os.listdir(vidpath)
    newgridlist=[]
    for vid in vidlist:
        if vid.endswith('.avi'):
     
            file = vid[:-4] + '.txt'
            #print(file,'read')
            gridarr=[]
            heatarr = []
            with open(labels_path+'/'+file, "r") as f:
            
                #print(file)
                content = f.readlines()
                for i in range(len(content)):
            
                    grid = np.zeros((4,4))
                    x,y = content[i].strip().split(',')[0:2]
                    coords = np.abs(([round(int(x)/20),round(int(y)/11.25)]))
                    grid[int(int(y)//180),int(int(x)//320)]=1
                    gridarr.append(grid)
                    heatarr.append(generate_hm(64, 64 ,coords,s=1.5))
            

            #print('heatarr:',len(heatarr))
            gdx = list(range(0,len(gridarr),n_frames))
       
            for i in range(len(gridarr)):
                try:
                    inpgrid.append( np.array( gridarr[gdx[i]:gdx[i+1]] ))
                    heatmap.append(np.array(heatarr[gdx[i]:gdx[i+1]]))
                    
                except IndexError:
                    pass
 
            
    frvids = pd.DataFrame([])
    frvids['vid'] = vlen
    frvids['labels'] = frlen
        
    #frvids.to_csv('labels_video.csv')
    return np.expand_dims(np.array(inpgrid),axis=-1),np.expand_dims(np.array(heatmap),axis=-1)


if __name__ == '__main__':
    
   
    inp_array = get_frames(vidpath='../gaze_data/val/videos')
    print('shape:',inp_array.shape)
    inp_grid,hm = get_grids(n_frames=16,labels_path='../gaze_data/val/labels',vidpath='../gaze_data/val/videos')
    print('heatmap = ',hm.shape,inp_grid.shape)
    
    #print(type(inp_array))
    #print(len(frames_array),len(frames_array[0]),len(frames_array[1]))
    #print(len(inp_grid))
    #print((inp_grid).shape)
    #print(frames_array[0].shape,frames_array[1].shape)
    