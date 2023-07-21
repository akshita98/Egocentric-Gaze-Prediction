import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from Inputs import *
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
import pickle



date = '21-8'
def ce_loss(y_pred,hm):
    hm = hm.squeeze()
    hm = torch.reshape((hm),(hm.shape[0]*hm.shape[1],-1))
    target = torch.argmax(hm,dim=1)
    y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    return nn.CrossEntropyLoss(reduction='sum')(pred,target)



def kl_loss(y_pred,hm):
    #print('shapes before',y_pred.shape,hm.shape)
    hm = torch.reshape((hm),(hm.shape[0]*hm.shape[1],-1))
    #y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2],-1))
    pred = F.log_softmax(pred,dim=1)
    #print('shapes',pred.shape,hm.shape)
    return nn.KLDivLoss(reduction = 'batchmean')(pred,hm)



def cap_acc(y_pred,y_true):
    y_true = torch.reshape((y_true),(y_true.shape[0]*y_true.shape[1],-1))
    y_true = torch.argmax(y_true,dim=1)
    y_pred = y_pred.squeeze()
    pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1],-1))
    y_pred = torch.argmax(pred,dim=1)
    
    correct = torch.eq(y_pred,y_true).cpu()
    correct = correct.numpy()
    return np.mean(correct)
    
    

def fgp_acc(y_pred,y_true):
    y_true = torch.reshape((y_true),(y_true.shape[0]*y_true.shape[1],y_true.shape[2],y_true.shape[3]))
    #y_pred = y_pred.squeeze()
    #print('shape',y_pred.shape)
    y_pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2],y_pred.shape[3],y_pred.shape[4]))
    x0_true = torch.argmax(torch.max(y_true,dim=-2)[0],dim=-1)
    y0_true = torch.argmax(torch.max(y_true,dim=-1)[0],dim=-1)
    x0_pred = torch.argmax(torch.max(y_pred,dim=-2)[0],dim=-1)
    y0_pred = torch.argmax(torch.max(y_pred,dim=-1)[0],dim=-1)
    #print(x0_true.dtype,x0_pred.dtype)
    ss = torch.square(x0_true-x0_pred,)+torch.square(y0_true-y0_pred)
    
    dist = torch.mean(torch.sqrt(ss.float()))
    return dist.item()
    

def box_plot_acc(y_pred,y_true):
    y_true = torch.reshape((y_true),(y_true.shape[0]*y_true.shape[1],y_true.shape[2],y_true.shape[3]))
    #y_pred = y_pred.squeeze()
    
    y_pred = torch.reshape((y_pred),(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2],y_pred.shape[3],y_pred.shape[4]))
    x0_true = torch.argmax(torch.max(y_true,dim=-2)[0],dim=-1)
    y0_true = torch.argmax(torch.max(y_true,dim=-1)[0],dim=-1)
    x0_pred = torch.argmax(torch.max(y_pred,dim=-2)[0],dim=-1)
    y0_pred = torch.argmax(torch.max(y_pred,dim=-1)[0],dim=-1)
    #print(x0_true.dtype,x0_pred.dtype)
    ss = torch.square(x0_true-x0_pred,)+torch.square(y0_true-y0_pred)
    
    dist = (torch.sqrt(ss.float())).cpu()
    return dist.numpy()
    
    
def generate_plots(df,d =date):
    history = pd.read_csv(df)
    plt.plot(history['val_loss'],label='val loss')
    plt.plot(history['train_loss'],label='train loss')
    plt.plot(history['test_loss'],label='test loss')
    plt.legend()
    plt.savefig(f'plots/'+f'final_loss({d})_plot'+'.png')
    plt.close()
    plt.plot(history['val_acc'],label='val deviation')
    plt.plot(history['train_acc'],label='train deviation')
    plt.plot(history['test_acc'],label='test deviation')
    plt.legend()
    plt.show()
    plt.savefig(f'plots/'+f'final_acc({d})_plot'+'.png')
    plt.close()
    print('done!')
    
    
    
    
batch=24
trainX = get_frames(vidpath='../gaze_data/train/videos',n_frames=2)
traincY,trainfY = get_grids(labels_path='../gaze_data/train/labels',vidpath='../gaze_data/train/videos',n_frames=2)

valX = get_frames(vidpath='../gaze_data/val/videos',n_frames=2)
valcY,valfY = get_grids(labels_path='../gaze_data/val/labels',vidpath='../gaze_data/val/videos',n_frames=2)

testX = get_frames(vidpath='../gaze_data/test/videos',n_frames=2)
testcY,testfY = get_grids(labels_path='../gaze_data/test/labels',vidpath='../gaze_data/test/videos',n_frames=2)
print('trainX=',trainX.shape)

trainX = np.rollaxis(trainX,-1,1)
valX = np.rollaxis(valX,-1,1)
testX = np.rollaxis(testX,-1,1)
print('trainX changed shape=',trainX.shape,trainfY.shape)

mean=[np.mean(trainX[:,0,:,:,:]),np.mean(trainX[:,1,:,:,:]),np.mean(trainX[:,2,:,:,:])]
std = [np.std(trainX[:,0,:,:,:]),np.std(trainX[:,1,:,:,:]),np.std(trainX[:,2,:,:,:])]

#trainX[:,0,:,:,:],trainX[:,1,:,:,:],trainX[:,2,:,:,:] = trainX[:,0,:,:,:]-mean[0],trainX[:,1,:,:,:]-mean[1],trainX[:,2,:,:,:]-mean[2]

#trainX[:,0,:,:,:],trainX[:,1,:,:,:],trainX[:,2,:,:,:] = trainX[:,0,:,:,:]/std[0],trainX[:,1,:,:,:]/std[1],trainX[:,2,:,:,:]/std[2]




#testX[:,0,:,:,:],testX[:,1,:,:,:],testX[:,2,:,:,:] = testX[:,0,:,:,:]-mean[0],testX[:,1,:,:,:]-mean[1],testX[:,2,:,:,:]-mean[2]

#testX[:,0,:,:,:],testX[:,1,:,:,:],testX[:,2,:,:,:] = testX[:,0,:,:,:]/std[0],testX[:,1,:,:,:]/std[1],testX[:,2,:,:,:]/std[2]



#valX[:,0,:,:,:],valX[:,1,:,:,:],valX[:,2,:,:,:] = valX[:,0,:,:,:]-mean[0],valX[:,1,:,:,:]-mean[1],valX[:,2,:,:,:]-mean[2]

#valX[:,0,:,:,:],valX[:,1,:,:,:],valX[:,2,:,:,:] = valX[:,0,:,:,:]/std[0],valX[:,1,:,:,:]/std[1],valX[:,2,:,:,:]/std[2]




train_datasetc = VidDataset(trainX,traincY)
val_datasetc = VidDataset(valX,valcY)
test_datasetc = VidDataset(testX,testcY)

print('normalised? :',np.max(trainX),np.min(trainX),np.mean(trainX[:,0,:,:,:]))
train_datasetf = VidDataset(trainX,trainfY)
val_datasetf = VidDataset(valX,valfY)
test_datasetf = VidDataset(testX,testfY)



train_loaderc = DataLoader(dataset = train_datasetc,batch_size=128,shuffle=True,num_workers=4)
val_loaderc = DataLoader(dataset=val_datasetc,batch_size=128,shuffle=True,num_workers=4)
test_loaderc = DataLoader(dataset=test_datasetc,batch_size=128,shuffle=True,num_workers=4)


train_loaderf = DataLoader(dataset = train_datasetf,batch_size=24,shuffle=True,num_workers=0)
val_loaderf = DataLoader(dataset=val_datasetf,batch_size=24,shuffle=True,num_workers=4)
test_loaderf = DataLoader(dataset=test_datasetf,batch_size=24,shuffle=True,num_workers=4)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"

device = torch.device("cuda")

model_fine = FineAP(3).to(device)

model_coarse = CoarseAP(3).to(device)


def train_coarse(model,epochs=100,ce_loss = ce_loss,tl=False):
    
    
    if tl:
        print('Transfer Learning')
        c3dict = torch.load('models/pytorch_models/c3d.pickle')
        modeldict = model.state_dict()
        pretrained_dict={}
        for i in c3dict.keys():
            for j in modeldict.keys():
                if i in j and 'deconv' not in j:
                    pretrained_dict[j] = c3dict[i]
      
      
        modeldict.update(pretrained_dict)
        model.load_state_dict(modeldict) 

        for par in model.named_parameters():
            if par[0] in pretrained_dict.keys():
                par[1].requires_grad=False
    
    
    history = pd.DataFrame([])
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7,8,9])
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9,weight_decay=1e-2)
    
    total_train_loss=[]
    total_val_loss=[]
    total_train_acc=[]
    total_val_acc=[]
    total_test_loss=[]
    total_test_acc=[]
    for e in range(epochs):
        
        loss_epoch = []
        acc_epoch = []
        model.train()
    
        for batch_idx, (train_X, train_Y) in enumerate(train_loaderc):
            train_X, train_Y = train_X.to(device,dtype = torch.float32), train_Y.to(device,dtype = torch.float32)
            optimizer.zero_grad()
            output = model(train_X)
            loss = ce_loss(output, train_Y)
            accuracy = cap_acc(output,train_Y)
            
            l2 = 0
            for p in model.named_parameters():
                if 'conv' and 'weight' in p[0]:
                    l2 = l2+ torch.pow(p[1].norm(2),2)
                    
            loss += (1e-6) * l2
            
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            acc_epoch.append(accuracy.item())
   
     
        total_train_loss.append(np.mean(loss_epoch))
        total_train_acc.append(np.mean(acc_epoch))
        print(f'Epcoh {e}: ,batch loss:{loss.item()},epoch loss:{np.mean(loss_epoch)},acc:{np.mean(acc_epoch)}',end=' ')
        
        model.eval()
        with torch.no_grad():
            val_loss_epoch=[]
            val_acc_epoch=[]
            for val_X, val_Y in val_loaderc:
                val_X,val_Y = val_X.to(device,dtype = torch.float32),val_Y.to(device,dtype = torch.float32)
                op = model(val_X)
                val_loss = ce_loss (op,val_Y)
                val_acc = cap_acc(op,val_Y)
                val_loss_epoch.append(val_loss.item())
                val_acc_epoch.append(val_acc.item())
             
            total_val_acc.append(np.mean(val_acc_epoch))   
            total_val_loss.append(np.mean(val_loss_epoch))
            print(f',epoch val_loss:{np.mean(val_loss_epoch)},val_acc={np.mean(val_acc_epoch)}')
            
            
            test_loss_epoch=[]
            test_acc_epoch=[]
            for test_X, test_Y in test_loaderc:
                test_X,test_Y = test_X.to(device,dtype = torch.float32),test_Y.to(device,dtype = torch.float32)
                op = model(test_X)
                test_loss = ce_loss(op,test_Y)
                test_acc = cap_acc(op,test_Y)
                test_loss_epoch.append(test_loss.item())
                test_acc_epoch.append(test_acc.item())
             
            total_test_acc.append(np.mean(test_acc_epoch))    
            total_test_loss.append(np.mean(test_loss_epoch))
            #print(f',epoch val_loss:{np.mean(test_loss_epoch)},val_acc={np.mean(test_acc_epoch)}')
            
    torch.save(model.state_dict(),f'models/pytorch_models/modelc_epoch_{epochs}.pth')
    history['train_loss'] = total_train_loss
    history['val_loss'] = total_val_loss
    history['train_acc'] = total_train_acc
    history['val_acc'] = total_val_acc
    history['test_acc'] = total_test_acc
    history['test_loss'] = total_test_loss
    history.to_csv('plots/'+f'history_coarse_{e+1}.csv')
    generate_plots(f'plots/history_coarse_{e+1}.csv')         
            
            
            
            
            
            
            
            
def train_fine(model,epochs=1,kl_loss = kl_loss,tl=False):
    history = pd.DataFrame([])
    min_val_loss = np.inf
    pat = 0
    pat_thresh=3
    box_df= pd.DataFrame([])
    test_box_df = pd.DataFrame([])
    
    if tl:
        print('Transfer Learning')
        c3dict = torch.load('models/pytorch_models/modelc_epoch_100.pth')
        modeldict = model.state_dict()
        pretrained_dict={}
        for i in c3dict.keys():
            for j in modeldict.keys():
                if i in j and 'deconv' not in j:
                    pretrained_dict[j] = c3dict[i]
      
      
        modeldict.update(pretrained_dict)
        model.load_state_dict(modeldict) 

        for par in model.named_parameters():
            if par[0] in pretrained_dict.keys():
                par[1].requires_grad=False
    
    
    
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7,8,9])
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9,weight_decay=1e-2)
    
    total_train_loss=[]
    total_val_loss=[]
    total_train_acc=[]
    total_val_acc=[]
    total_test_acc=[]
    total_test_loss=[]
    print('Training started')
    for e in range(epochs):
        loss_epoch = []
        acc_epoch = []
        box_plot=[]
        test_box_plot=[]
        model.train()
    
        for batch_idx, (train_X, train_Y) in enumerate(train_loaderf):
            train_X, train_Y = train_X.to(device,dtype = torch.float32), train_Y.to(device,dtype = torch.float32)
            #print((train_X).dtype)
            optimizer.zero_grad()
            output = model(train_X)
            loss = kl_loss(output, train_Y)
            l2 = 0
            for p in model.named_parameters():
                if 'conv' and 'weight' in p[0]:
                    l2 = l2+ torch.pow(p[1].norm(2),2)
            loss = loss+l2*(1e-6)
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            accuracy = fgp_acc(output,train_Y)
            acc_epoch.append(accuracy)
        
        
   
     
        
        total_train_loss.append(np.mean(loss_epoch))
        total_train_acc.append(np.mean(acc_epoch))
       
        print(f'Epcoh {e}/{epochs}: ,batch loss:{loss.item()},epoch loss:{np.mean(loss_epoch)},acc:{np.mean(acc_epoch)}',end=' ')   
    

        model.eval()
        with torch.no_grad():
            val_loss_epoch=[]
            val_acc_epoch=[]
            for val_X, val_Y in val_loaderf:
                val_X,val_Y = val_X.to(device,dtype = torch.float32),val_Y.to(device,dtype = torch.float32)
                op = model(val_X)
                val_loss = kl_loss(op,val_Y)
                val_acc = fgp_acc(op,val_Y)
                box_plot.extend(list(box_plot_acc(op,val_Y)))
             
                
                val_loss_epoch.append(val_loss.item())
                val_acc_epoch.append(val_acc)
            
            
            
            total_val_acc.append(np.mean(val_acc_epoch))    
            total_val_loss.append(np.mean(val_loss_epoch))
            
            #if total_val_loss[e]>min_val_loss:
            #    pat+=1
                
            #    if pat>pat_thresh:
            #        print('/nEarly Stopping at epoch=',early_e)
            #        print(f'model saved for {early_e+1} epochs')
                    
            #        history['train_loss'] = pd.Series(total_train_loss)
            #        history['val_loss'] = pd.Series(total_val_loss)
            #        history['train_acc'] = pd.Series(total_train_acc)
            #        history['val_acc'] = pd.Series(total_val_acc)
            #        history['test_loss'] = pd.Series(total_test_loss)
            #        history['test_acc'] = pd.Series(total_test_acc)
            #        history.to_csv(f'plots/'+f'history_fine_norm1({date}).csv')
            #        generate_plots(f'plots/history_fine_norm1({date}).csv')
            #        return None
                
                            
                    
            #else:
            #    min_val_loss = total_val_loss[e]
            #    early_e = e
            #    torch.save(model.state_dict(),f'models/pytorch_models/modelf_({date}).pth')
                
            
            
            
            print(f',epoch val_loss:{np.mean(val_loss_epoch)},val_acc={np.mean(val_acc_epoch)}')
            
            
            test_loss_epoch=[]
            test_acc_epoch=[]
            for test_X, test_Y in test_loaderf:
                test_X,test_Y = test_X.to(device,dtype = torch.float32),test_Y.to(device,dtype = torch.float32)
                op = model(test_X)
                test_loss = kl_loss(op,test_Y)
                test_acc = fgp_acc(op,test_Y)
                test_loss_epoch.append(test_loss.item())
                test_acc_epoch.append(test_acc)
                test_box_plot.extend(list(box_plot_acc(op,test_Y)))

            total_test_acc.append(np.mean(test_acc_epoch))    
            total_test_loss.append(np.mean(test_loss_epoch))
            
        
        box_df[f'{e}'] = box_plot
        test_box_df[f'{e}'] = test_box_plot
    print(f'completed {epochs} epochs')
    torch.save(model.state_dict(),f'models/pytorch_models/modelf_({date}).pth')
    history['train_loss'] = total_train_loss
    history['val_loss'] = total_val_loss
    history['train_acc'] = total_train_acc
    history['val_acc'] = total_val_acc
    history['test_loss'] = total_test_loss
    history['test_acc'] = total_test_acc
    history.to_csv('plots/'+f'history_fine({date}).csv')
    generate_plots(f'plots/history_fine({date}).csv')
   # box_df.to_csv(f'val_box_plots({date}).csv')
   # test_box_df.to_csv(f'test_box_plots({date}).csv')

    
    


    
if __name__=='__main__':
    
    s = time.time()
    #train_coarse(model_coarse,epochs=100,tl=True)
    train_fine(model_fine,epochs=25,tl=True)
    #generate_plots(f'plots/history_fine_norm_25.csv')
    e = time.time()
    print('it took',e-s)
    