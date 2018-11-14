# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:35:06 2018

@author: alexa
"""

import math, time, torch, scipy
import numpy as np
import matplotlib.pyplot as plt

class batchify:
    def __init__(self, data, finetune=False, bsz=64, training=False):
        if finetune:
            self.data_2 = data[0]
            self.data   = data[1]
        else:
            self.data = data
        self.bsz = bsz
        self.start = 0
        
        self.finetune= finetune
        self.end = bsz
        self.training = training
        
        self.length = len(self.data)//self.bsz
        
        if len(self.data)%self.bsz != 0:
            self.length += 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):     
        if index >= self.length:
            self.start = 0
            self.end = self.bsz
            raise IndexError
        batch = self.data[self.start:self.end].astype(np.float32)
        self.start += self.bsz
        self.end += self.bsz
        if self.end > len(self.data) - 1:
            self.end = -1
         
        if self.finetune:
            users = torch.tensor(batch[:,0].astype(int))
            items = torch.tensor(batch[:,1].astype(int))
            ratings = torch.tensor(batch[:,2].astype(np.float32), requires_grad = self.training)
            itemsFeature = torch.tensor(self.data_2[batch[:,1].astype(int), :].astype(np.float32), requires_grad = self.training)
            input_variable = (itemsFeature, users, items)
            target_variable = ratings
        else:
            input_variable = torch.tensor(batch, requires_grad = self.training)
            target_variable = torch.tensor(batch, requires_grad = self.training)
        return (input_variable, target_variable)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (-%s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%3.fm %2.fs' % (m, s)

def save_model(name,encoder,decoder,iteration,local=False, models_path='./'):
    with open("{}{}_iter.txt".format(models_path, name), 'w') as f:
        f.write(str(iteration))
    torch.save(encoder.state_dict(),"{}{}_encoder.pt".format(models_path, name))
    torch.save(decoder.state_dict(),"{}{}_decoder.pt".format(models_path, name))

def load_model(fname,model, models_path='./'):
    loaded_state_dict = torch.load(models_path+''+fname, map_location=None if use_cuda else {'cuda:0':'cpu'})
    state_dict = model.state_dict()
    state_dict.update(loaded_state_dict)
    model.load_state_dict(loaded_state_dict)

def load_loss(name, loss_type, models_path='./'): 
    losses = []
    try:
        with open("{}{}_{}_losses.txt".format(models_path, name, loss_type), 'r') as f:
            for line in f:
                losses = [float(value) for value in line.split(";")]
                break
    except:
        print('No loss file')
    return losses

def save_loss(name, loss_type, losses, models_path='./'):
    with open("{}{}_{}_losses.txt".format(models_path, name, loss_type), 'w') as f:
        f.write(';'.join([str(value) for value in losses]))

def load_iter(name, models_path='./'):
    try:
        with open("{}{}_iter.txt".format(models_path, name), 'r') as f:
            for line in f:
                start_iter = int(line)
                break
    except:
        start_iter = 1
        print('error during downloading of file')
    return start_iter

def log_func(x, a, b, c):
    return a*x**2+b*x+c

def showPlot(points, interpol=False):
    fig, ax = plt.subplots(figsize=(20,15))
    # this locator puts ticks at regular intervals
    interval = (max(points)-min(points))/20
    loc = ticker.MultipleLocator(base=interval)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    if interpol:
        x = np.arange(1e-6, len(points[15:]))
        y = points[15:]
        popt, pcov = scipy.optimize.curve_fit(log_func, x, y, p0=(1, 1, 1))
        xx = np.linspace(1e-6, int(max(x)*1.5), 500)
        yy = log_func(xx, *popt)
        yy[0] = y[0]
        plt.plot(xx,yy)
    plt.grid()
    plt.show()