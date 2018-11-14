import torch, pickle, os, csv
from torch.utils.data import Dataset
use_cuda = torch.cuda.is_available()
import numpy as np

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
        encoding = 'utf-8'
        self.chunkSize = 1e5
        self.folder_path = '.'.join(ratings_path.split('.')[:-1])
        
        with open(movies_data_path, 'rb') as file:
            raw_data = pickle.load(file)
            self.movies_data = raw_data['data']
            self.movies_ids = dict([(id, i) for i, id in enumerate(raw_data['ids'])])
        
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
            self.currentPart = 0
            dataToSave = []
            with open(ratings_path, 'r', encoding=encoding, newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for i, line in enumerate(reader):
                    if int(line[1]) in self.movies_ids:
                        if len(dataToSave) < self.chunkSize:
                            dataToSave.append((int(line[0]),int(line[1]), float(line[2])))
                        else:
                            with open(self.folder_path+'/chunk_'+str(self.currentPart)+'.npy', 'wb') as new_file:
                                pickle.dump(dataToSave, new_file) 
                            dataToSave = []
                            self.currentPart += 1
            if len(dataToSave) != 0:
                with open(self.folder_path+'/chunk_'+str(self.currentPart)+'.npy', 'wb') as new_file:
                    pickle.dump(dataToSave, new_file)
            self.length = i
        else:
            files = os.listdir(self.folder_path)
            files_nb = sorted([int(name.split('_')[1].split('.')[0]) for name in files])
            with open(self.folder_path+'/chunk_'+str(files_nb[-1])+'.npy', 'rb') as pickler:
                self.length = int(len(pickle.load(pickler)) + (len(files) - 1) * self.chunkSize - 1)
        
        self.currentPart = 0
        with open(self.folder_path+'/chunk_'+str(self.currentPart)+'.npy', 'rb') as pickler:
            self.currentData = pickle.load(pickler)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
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
                    
if __name__ == '__main__':    
    transformed_dataset = finetuneDataset('./data/ratings.csv', './data/data.npy')
    