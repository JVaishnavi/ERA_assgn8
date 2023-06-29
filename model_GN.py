#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:48:27 2023

@author: vaishnavijanakiraman
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dropout_value = 0.1
        
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8, 16),
            nn.Dropout(dropout_value)
        ) #IP: 32, OP: 32
        
        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8, 24),
            nn.Dropout(dropout_value)
        ) # IP: 32, OP: 32
        
        
        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8, 8),
            nn.Dropout(dropout_value)
        ) # IP: 32, OP: 32
        
        
        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # IP: 32, OP: 16
        

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(8, 16),
            nn.Dropout(dropout_value)
        ) # IP: 16, OP: 16
        

        # CONVOLUTION BLOCK 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(8, 32),
            nn.Dropout(dropout_value)
        ) # IP: 16, OP: 16
        
        
        # CONVOLUTION BLOCK 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(16, 48),
            nn.Dropout(dropout_value)
        ) # IP: 16, OP: 16
    
    
        # CONVOLUTION BLOCK 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(10, 10),
            nn.Dropout(dropout_value)
        ) # IP: 16, OP: 16
        
        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # IP: 16, OP: 8
        
        
        # CONVOLUTION BLOCK 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(16, 16),
            nn.Dropout(dropout_value)
        ) # IP: 8, OP: 8
        

        # CONVOLUTION BLOCK 9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(16, 32),
            nn.Dropout(dropout_value)
        ) # IP: 8, OP: 6
        
        
        # CONVOLUTION BLOCK 10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.GroupNorm(32, 64),
            nn.Dropout(dropout_value)
        ) # IP: 6, OP: 4
    
        # GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # IP: 4 x 4 x 256 , OP: 1 x 1 x 256

        # CONVOLUTION BLOCK 11
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        # IP: 1 x 1 x 256, OP: 1 x 1 x 10

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)        
        x = self.convblock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)