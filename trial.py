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
date = '1-1'
def generate_plots(df,d =date):
    history = pd.read_csv(df)
    plt.plot(history['val_loss'],label='val loss')
    plt.plot(history['train_loss'],label='train loss')
    plt.plot(history['test_loss'],label='test loss')
    plt.legend()
    plt.savefig(f'final_loss({d})_plot'+'.png')
    plt.close()
    plt.plot(history['val_acc'],label='val deviation')
    plt.plot(history['train_acc'],label='train deviation')
    plt.plot(history['test_acc'],label='test deviation')
    plt.legend()
    plt.show()
    plt.savefig(f'final_acc({d})_plot'+'.png')
    plt.close()
    print('done!')
    
df = 'plots/history_fine(28-7).csv' 
model_path = 'models/pytorch_models/modelf_full_epoch(22-7).pth'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
device = torch.device("cuda")
fine_mod = FineAP(3).to(device)
fine_mod = nn.DataParallel(fine_mod, device_ids=[0,1,2,3,4,5,6,7,8,9])
fine_mod.load_state_dict(torch.load(model_path))

for param in fine_mod.parameters():
    print('shape',param.shape)
input_s =torch.randn(1,3,16,64,64)

print('out',fine_mod(input_s).shape)