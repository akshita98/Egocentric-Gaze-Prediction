
import sys
sys.path.insert(1, '/flask/')

import io
import json
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

from pytorch_model import *
from collections import OrderedDict
import datetime
from datetime import datetime
import io
import json
import numpy as np
import cv2
from PIL import Image


def InferenceEyegaze(imagearr,imageindex,model):
    results=[0,0]
    
    torch.cuda.synchronize()
    inp_array = np.array([imagearr[imageindex:imageindex+16]])
    
    # shape = (16, 64, 64, 3)
    # mark
    inp_array = np.rollaxis(inp_array, -1, 1)
    
    assert inp_array.shape==(1,3,16,64,64),'input array should have shape=(1,3,16,64,64)'
    #predictions = model(inp_array)
    
    # mark
    inp_array = torch.tensor(inp_array, dtype=torch.float32, requires_grad=False)
    
    # mark
    with torch.no_grad():
        torch.cuda.synchronize()
        
        t1=datetime.now()
        p = model(inp_array)
        
        t2=datetime.now()        
        torch.cuda.synchronize()
        t3=datetime.now()
        #predictions=p.squeeze().cpu().numpy()        
        
        p=p.squeeze()        
        p=p.cpu()
        predictions=p[-1].numpy()
        
        torch.cuda.synchronize()
        t4=datetime.now()
        torch.cuda.synchronize()
    
    frame=predictions
    nx = np.argmax((np.max(frame,axis=1)))
    ny = np.argmax((np.max(frame,axis=0)))
    x = int(round(ny*20))
    y = int(round(nx*11.25))
    results =(x,y)
    
    #print((t2-t1).total_seconds(),(t3-t2).total_seconds(),(t4-t3).total_seconds())
      
    return results,(t2-t1).total_seconds(),(t3-t2).total_seconds(),(t4-t3).total_seconds()

def loadmodel(model_path):
    device = torch.device("cuda")
    fine_mod = FineAP(3).to(device)
    fine_mod = nn.DataParallel(fine_mod)
    fine_mod.load_state_dict(torch.load(model_path))
    
    return fine_mod


def loadimages(folder='preimages/',imgcount=1970, n_startindex=0):
    frames = []
    inp = []
    for i in range(n_startindex, n_startindex+imgcount):
        filename = "{0}/{1}.bmp".format(folder, i)
        img = cv2.imread(filename, 0)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.destroyAllWindows()
    return frames,imgcount

def main():
    model_path='models/pytorch_models/modelf_epoch_99.pth'
    imagefolder='../Tianzi/data/Camera3_taskA_trial4_1573164947973_ori/'
    
    model = loadmodel(model_path)
    imagearr,imgcount=loadimages(folder=imagefolder)
    print(imgcount)
    res_arr=[]
    #imgcount=20
    for i in range(0,imgcount-16):
        t1=datetime.now()
        res,modelgap,syngap,cpugap=InferenceEyegaze(imagearr ,i,model)
        t2=datetime.now()
        print("t1",t1,"index",i,"res",res,"modelgap", modelgap,"synchronizegap",syngap,"cpugap",cpugap,"totalgap",(t2-t1).total_seconds())
        res_arr.append([i,res[0],res[1],modelgap,syngap,cpugap,(t2-t1).total_seconds()])
    
    with open("speed_tensorserve.csv", "w") as f:
        f.write("index,resx,resy,modelgap,synchronizegap,cpugap,totalgap\n")
        for item in res_arr:
            f.write(', '.join(map(str, item))+"\n")
    return

if __name__ == '__main__':
    main()
