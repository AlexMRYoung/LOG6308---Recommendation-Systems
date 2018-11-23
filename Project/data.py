import torch, pickle, os
from torch.utils.data import Dataset
use_cuda = torch.cuda.is_available()
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class pretrainDataset:
    def __init__(self, file_path, training=True):
        
        with open(file_path, 'rb') as file:
            self.data = pickle.load(file)['data'].astype(np.float32)
        self.data = self.data[:int(self.data.shape[0]*0.8)] if training else self.data[int(self.data.shape[0]*0.8):]
                
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        
        return (self.data.getrow(index).toarray().reshape(-1), self.data.getrow(index).toarray().reshape(-1))

class finetuneDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, ratings_path, movies_data_path, training=True):
        assert os.path.isfile(ratings_path)
        # Loading data
        self.folder_path = '.'.join(ratings_path.split('.')[:-1])
        self.chunkSize = 1e5
        self.offset = 0
        
        with open(movies_data_path, 'rb') as file:
            raw_data = pickle.load(file)
            self.movies_data = raw_data['data']
            self.movies_ids = dict([(id, i) for i, id in enumerate(raw_data['ids'])])
        
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
            train_folder_path = self.folder_path+'/train/'
            valid_folder_path = self.folder_path+'/valid/'
            os.mkdir(train_folder_path)
            os.mkdir(valid_folder_path)
            
            ratings_df = pd.read_csv(ratings_path)
            ratings_df = shuffle(ratings_df).reset_index(drop=True)
            input_size = self.movies_data.shape[1]
            nb_users = ratings_df['userId'].max()
            
            with open(self.folder_path+'/params.pkl', 'wb') as pickler:
                pickle.dump((int(input_size), int(nb_users), 46000), pickler)
                
            while self.chunkSize > ratings_df.shape[0] * 0.8:
                self.chunkSize /= 2
            self.currentPart = 0
            dataToSave = []
            for i, line in ratings_df.iterrows():
                if int(line['movieId']) in self.movies_ids:
                    if i < ratings_df.shape[0] * 0.8:
                        subfolder_path = train_folder_path  
                    else:
                        if subfolder_path == train_folder_path:
                            with open(subfolder_path+'chunk_'+str(self.currentPart)+'.npy', 'wb') as new_file:
                                pickle.dump(dataToSave, new_file) 
                                dataToSave = []
                        subfolder_path = valid_folder_path
                        self.currentPart = 0
                    if len(dataToSave) < self.chunkSize:
                        dataToSave.append((int(line['userId']),int(line['movieId']), float(line['rating'])))
                    else:
                        with open(subfolder_path+'chunk_'+str(self.currentPart)+'.npy', 'wb') as new_file:
                            pickle.dump(dataToSave, new_file) 
                        dataToSave = []
                        self.currentPart += 1
            if len(dataToSave) != 0:
                print(len(dataToSave))
                with open(subfolder_path+'chunk_'+str(self.currentPart)+'.npy', 'wb') as new_file:
                    pickle.dump(dataToSave, new_file)
        
        self.folder_path += '/train' if training else '/valid'
        files = os.listdir(self.folder_path)
        files_nb = sorted([int(name.split('_')[1].split('.')[0]) for name in files])
        if len(files) > 1:
            with open(self.folder_path+'/chunk_'+str(files_nb[0])+'.npy', 'rb') as pickler:
                self.chunkSize = int(len(pickle.load(pickler)))
        
        with open(self.folder_path+'/chunk_'+str(files_nb[-1])+'.npy', 'rb') as pickler:
            self.length = int(len(pickle.load(pickler)) + (len(files) - 1) * self.chunkSize )
        
        self.currentPart = 0
        with open(self.folder_path+'/chunk_'+str(self.currentPart)+'.npy', 'rb') as pickler:
            self.currentData = pickle.load(pickler)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx += self.offset
        # go to next file
        if idx // self.chunkSize != self.currentPart:
            self.currentPart = int(idx // self.chunkSize)
            with open(self.folder_path+'/chunk_'+str(self.currentPart)+'.npy', 'rb') as pickler:
                self.currentData = pickle.load(pickler)
        i = int(idx - self.currentPart * self.chunkSize)
        
        userID = self.currentData[i][0] - 1
        movieID = self.movies_ids[self.currentData[i][1]]
        movieData = self.movies_data[movieID].toarray().astype(np.float32).reshape(-1)
        return ((movieData, userID, movieID), np.float32(self.currentData[i][2]))
    
    def resume(self, offset):
        self.offset = offset
        self.currentPart = int(offset // self.chunkSize)
        with open(self.folder_path+'/chunk_'+str(self.currentPart)+'.npy', 'rb') as pickler:
            self.currentData = pickle.load(pickler)
                    
if __name__ == '__main__': 
    finetuneDataset("./data/ratings_small.csv", "./data/data.npy", training=False)
    
    print(len(finetuneDataset("./data/ratings_small.csv", "./data/data.npy", training=False)))
    print(len(finetuneDataset("./data/ratings_small.csv", "./data/data.npy", training=True)))
    