import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import shutil
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import random
import time

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


class opt:
    batch_size=64
    workers=2
    cuda=torch.cuda.is_available()
    lr=0.001
    normal_train_path="../input/ecg-data/normal_train.csv"
    normal_valid_path="../input/ecg-data/normal_valid.csv"
    patient_valid_path="../input/ecg-data/patient_valid.csv"
    
device = torch.device("cuda:0" if opt.cuda else "cpu")

class Autoencoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #conv1d(in_channel,out_channel,kernel_size,stride,padding,bias=True,)
        #input=batch_size*187*1
        self.Encoder=nn.Sequential(
            
            nn.Conv1d(1,2,8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*90*2
            
            nn.Conv1d(2,4,11),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*40*4
            
            nn.Conv1d(4,8,11),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*15*4
            
            nn.Conv1d(8,16,6),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*5*4
            
            nn.Conv1d(16,32,5),
            nn.ReLU(inplace=True),#batch_size*1*32
        )
        
        #convtranspose1d(in_channel,out_channel,kernel_size,stride,padding,output_padding,bias=True)
        #input=batch_size*1*32
        self.Decoder=nn.Sequential(
            
            nn.ConvTranspose1d(32,16,15,1,0),#batch_size*15*16
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(16,8,12,2,0),#batch_size*40*8
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(8,4,12,2,0),#batch_size*90*4
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(4,2,12,2,5),#batch_size*180*2
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(2,1,8,1,0),#batch_size*180*1
            nn.Sigmoid(),
        )
        
    def forward(self,inp):
        inp=inp.view(-1,1,187)
        latent_vector=self.Encoder(inp)
        fake_signal=self.Decoder(latent_vector)
        
        return fake_signal.view(-1,187)
    
    
class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        #input=(seq_len,batch_input,input_size)(187,64,1)
        #lstm=(input_size,hidden_size)
        self.hidden_size=20
        self.lstm1 = nn.LSTMCell(1,self.hidden_size)
        self.lstm2 = nn.LSTMCell(1,self.hidden_size)
        self.linear = nn.Linear(self.hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def Encoder(self,inp):
        ht=torch.zeros(inp.size(0),self.hidden_size,dtype=torch.float,device=device)
        ct=torch.zeros(inp.size(0),self.hidden_size,dtype=torch.float,device=device)
        
        for input_t in inp.chunk(inp.size(1),dim=1):
            ht,ct=self.lstm1(input_t,(ht,ct))
            
        return ht,ct
    
    def Decoder(self,ht,ct):
        
        ot=torch.zeros(ht.size(0),1,dtype=torch.float,device=device)
        outputs=torch.zeros(ht.size(0),187,dtype=torch.float,device=device)
        
        for i in range(187):
            ht,ct=self.lstm2(ot,(ht,ct))
            ot=self.sigmoid(self.linear(ht))
            outputs[:,i]=ot.squeeze()
            
        return outputs
        
    def forward(self,inp):
        
        he,ce=self.Encoder(inp) #hidden encoder,cell_state encoder
        out=self.Decoder(he,ce)
    
        return torch.flip(out,dims=[1])
    
class KLAutoencoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #conv1d(in_channel,out_channel,kernel_size,stride,padding,bias=True,)
        #input=batch_size*187*1
        self.Encoder=nn.Sequential(
            
            nn.Conv1d(1,2,8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*90*2
            
            nn.Conv1d(2,4,11),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*40*4
            
            nn.Conv1d(4,8,11),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*15*4
            
            nn.Conv1d(8,16,6),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),#batch_size*5*4
            
            nn.Conv1d(16,32,5),
            nn.Softmax(dim=1),#batch_size*1*32
        )
        
        #convtranspose1d(in_channel,out_channel,kernel_size,stride,padding,output_padding,bias=True)
        #input=batch_size*1*32
        self.Decoder=nn.Sequential(
            
            nn.ConvTranspose1d(32,16,15,1,0),#batch_size*15*16
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(16,8,12,2,0),#batch_size*40*8
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(8,4,12,2,0),#batch_size*90*4
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(4,2,12,2,5),#batch_size*180*2
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(2,1,8,1,0),#batch_size*180*1
            nn.Sigmoid(),
        )
        
    def forward(self,inp):
        inp=inp.view(-1,1,187)
        latent_vector=self.Encoder(inp)
        fake_signal=self.Decoder(latent_vector)
        
        return fake_signal.view(-1,187),torch.log(latent_vector.view(-1,32))

class KLLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        #input=(seq_len,batch_input,input_size)(187,64,1)
        #lstm=(input_size,hidden_size)
        self.hidden_size=20
        self.lstm1 = nn.LSTMCell(1,self.hidden_size)
        self.lstm2 = nn.LSTMCell(1,self.hidden_size)
        self.linear = nn.Linear(self.hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def Encoder(self,inp):
        ht=torch.zeros(inp.size(0),self.hidden_size,dtype=torch.float,device=device)
        ct=torch.zeros(inp.size(0),self.hidden_size,dtype=torch.float,device=device)
        
        for input_t in inp.chunk(inp.size(1),dim=1):
            ht,ct=self.lstm1(input_t,(ht,ct))
            
        return ht,ct
    
    def Decoder(self,ht,ct):
        
        ot=torch.zeros(ht.size(0),1,dtype=torch.float,device=device)
        outputs=torch.zeros(ht.size(0),187,dtype=torch.float,device=device)
        
        for i in range(187):
            ht,ct=self.lstm2(ot,(ht,ct))
            ot=self.sigmoid(self.linear(ht))
            outputs[:,i]=ot.squeeze()
            
        return outputs
        
    def forward(self,inp):
        
        he,ce=self.Encoder(inp) #hidden encoder,cell_state encoder
        out=self.Decoder(he,ce)
        return torch.flip(out,dims=[1]),torch.log(F.softmax(he,dim=1))

class MLPAutoencoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #conv1d(in_channel,out_channel,kernel_size,stride,padding,bias=True,)
        #input=batch_size*187*1
        self.Encoder=nn.Sequential(
            
            nn.Linear(187,500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500,100),
            nn.ReLU(inplace=True),
            
            nn.Linear(100,32),
            nn.Softmax(dim=1),
        )
        
        #convtranspose1d(in_channel,out_channel,kernel_size,stride,padding,output_padding,bias=True)
        #input=batch_size*1*32
        self.Decoder=nn.Sequential(
            
            nn.Linear(32,100),
            nn.ReLU(inplace=True),
            
            nn.Linear(100,500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500,187),
            nn.Sigmoid()
        )
        
    def forward(self,inp):
        inp=inp.view(-1,187)
        latent_vector=self.Encoder(inp)
        fake_signal=self.Decoder(latent_vector)
        
        return fake_signal.view(-1,187)    
    
class KLMLPAutoencoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #conv1d(in_channel,out_channel,kernel_size,stride,padding,bias=True,)
        #input=batch_size*187*1
        self.Encoder=nn.Sequential(
            
            nn.Linear(187,500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500,100),
            nn.ReLU(inplace=True),
            
            nn.Linear(100,32),
            nn.Softmax(dim=1),
        )
        
        #convtranspose1d(in_channel,out_channel,kernel_size,stride,padding,output_padding,bias=True)
        #input=batch_size*1*32
        self.Decoder=nn.Sequential(
            
            nn.Linear(32,100),
            nn.ReLU(inplace=True),
            
            nn.Linear(100,500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500,187),
            nn.Sigmoid()
        )
        
    def forward(self,inp):
        inp=inp.view(-1,187)
        latent_vector=self.Encoder(inp)
        fake_signal=self.Decoder(latent_vector)
        
        return fake_signal.view(-1,187),torch.log(latent_vector.view(-1,32))    
    

def loss_function(pred,real):
    #criterion=nn.MSELoss(reduction="sum")
    mse=F.mse_loss(pred,real,reduction="sum")
    return mse


def loss_function_kl(out,real):
    pred,latent=out[0],out[1]
    z=torch.rand(latent.size(0), latent.size(1),device=device)
    
    mse=F.mse_loss(pred,real,reduction="sum")
    kld=F.kl_div(latent,z,reduction="batchmean")
    
    return mse