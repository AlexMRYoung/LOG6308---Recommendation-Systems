import numpy as np
from utils.helpers import batchify, timeSince
from utils.tokenizer import tokenize_corpus
from model import CDL
import torch, pickle, argparse, os, time, random
from torch.utils.data import Dataset, DataLoader


class dataSet(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, transform=None):
        # Loading data
        encoding = 'utf-8'
        path_to_data = './data/'
        path = path_to_data+"dataEmbeded.pkl"

        if not os.path.isfile(path_to_data+'processed_data0.npy'):
            with open(path, 'rb') as pickler:
                data = pickle.load(pickler)
            processed_data = tokenize_corpus(data[:,0], stop_words = False, BoW = True)
            for i in range(10):
                with open(path_to_data+'processed_data'+str(i)+'.npy', 'wb') as file:
                    pickle.dump(processed_data[len(processed_data)//10*i:len(processed_data)//10*(i+1),], file)
        
        self.currentPart = 0
        with open(path_to_data+'processed_data'+self.currentPart+'.npy', 'rb') as pickler:
                self.currentData = pickle.load(pickler)
        self.currentDataChosen = 0

    def __len__(self):
        return len(self.currentData)*10

    def __getitem__(self, idx):
        # middle of a file
        if self.currentDataChosen < len(self.currentData):
            return self.currentData[self.currentDataChosen,]
        # go to next file
        else if self.currentPart < 9:
            self.currentPart += 1
            with open(path_to_data+'processed_data'+self.currentPart+'.npy', 'rb') as pickler:
                self.currentData = pickle.load(pickler)
            self.currentDataChosen = 0
            return self.currentData[self.currentDataChosen,]
        # get back to the first file
        self.currentPart = 0
        with open(path_to_data+'processed_data'+self.currentPart+'.npy', 'rb') as pickler:
            self.currentData = pickle.load(pickler)
        self.currentDataChosen = 0
        return self.currentData[self.currentDataChosen,]