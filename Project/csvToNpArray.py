import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer
import pickle as pkl
from utils.tokenizer import tokenize_corpus

dataEmbeded = np.empty((45466,10), dtype=object)
ids = []

def getNames(data):
    names = []
    if not data:
        return names
    parsedData = eval(data)
    if not parsedData:
        return names
    for pieceOfInfo in parsedData:
        name = pieceOfInfo['name']
        names.append(name)
    return np.array(names)

with open('./data/links.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)
    id_to_movieId = dict()
    for line in reader:
        try:
            id_to_movieId[int(line[2])] = int(line[0])
        except:
            pass
print(len(id_to_movieId))
notIn=[]

with open('./data/movies_metadata.csv', encoding= 'utf-8') as csvFile:
    reader = csv.DictReader(csvFile)
    i = 0
    for row in reader:
        dataEmbeded[i, 0] = row['overview']
        try:
            dataEmbeded[i, 1] = id_to_movieId[int(row['id'])]
        except:
            notIn.append(row['id'])
            pass
        dataEmbeded[i, 2] = row['adult'] == 1
        dataEmbeded[i, 3] = row['budget']
        dataEmbeded[i, 4] = getNames(row['genres'])
        dataEmbeded[i, 5] = row['popularity']
        dataEmbeded[i, 6] = getNames(row['production_companies'])
        dataEmbeded[i, 7] = row['production_countries'] == "[{'iso_3166_1': 'US', 'name': 'United States of America'}]"
        dataEmbeded[i, 8] = row['revenue']
        dataEmbeded[i, 9] = getNames(row['spoken_languages'])
        i += 1

one_hot = MultiLabelBinarizer()
genres = one_hot.fit_transform(dataEmbeded[:,4])
production_companies = one_hot.fit_transform(dataEmbeded[:,6])
spoken_languages = one_hot.fit_transform(dataEmbeded[:,9])
BoW = tokenize_corpus(dataEmbeded[:,0], stop_words = False, BoW = True)



dataEmbeded = np.delete(dataEmbeded, [4, 6, 9], 1)
dataEmbeded = np.c_[dataEmbeded,genres]
dataEmbeded = np.c_[dataEmbeded,spoken_languages]

## production_companies too large !
#dataEmbeded = np.c_[dataEmbeded,production_companies]

#U,s,V = np.linalg.svd(production_companies, full_matrices=False) 

# keep 20 first singular values
#dataEmbeded = np.c_[dataEmbeded, U[:,20]]

# saving
with open('./data/dataEmbeded.pkl', 'wb') as pikeler:
    pkl.dump(dataEmbeded, pikeler)
    
with open('./data/data.npy', 'wb') as pikeler:
    data = {'ids':dataEmbeded[:, 1], 'data':BoW}
    pkl.dump(data, pikeler)

""" get the data back
with open('dataEmbeded.pkl', 'rb') as pikeler:
    dataEmbeded = cPickle.load(pikeler)
"""

