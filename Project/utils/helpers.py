# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:35:06 2018

@author: alexa
"""

import math, time, scipy, pickle
import numpy as np
import matplotlib.pyplot as plt

def stringArgToList(string):
    out = string[1:-1].replace(' ', '').split(',')
    return [int(arg) for arg in out]

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

def load_loss(name, loss_type, models_path='./'): 
    try:
        with open("{}{}_{}_losses.pkl".format(models_path, name, loss_type), 'rb') as f:
            losses = pickle.load(f)
    except:
        losses = []
    return losses

def save_loss(loss, name, loss_type, models_path='./'):
    losses = load_loss(name, loss_type, models_path)
    losses.append(loss)
    with open("{}{}_{}_losses.pkl".format(models_path, name, loss_type), 'wb') as f:
        pickle.dump(losses,f)

def log_func(x, a, b, c):
    return a*x**2+b*x+c

def savePlot(loss_name, path, name):
    train_loss = load_loss(name, 'train', models_path=path)
    valid_loss = load_loss(name, 'valid', models_path=path)
    valid_loss = [train_loss[0]] + valid_loss
    
    x1 = np.linspace(0, len(train_loss), len(train_loss))
    x2 = np.linspace(0, len(train_loss), len(valid_loss))
    
    plt.plot(x1, train_loss)
    plt.plot(x2, valid_loss)
    plt.xlabel('Batch number')
    plt.ylabel(loss_name)
    plt.legend(['Training', 'Validation'])
    plt.grid()
    plt.savefig(path + name + '_losses.png')