import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer
import cPickle
from sklearn import decomposition
import matplotlib


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

def getFloat(row, st):
    if row[st]:
        try:
            return float(row[st]) 
            pass
        except:
            return 0
            pass
    else:
        return 0

dataEmbeded = np.empty((45466,9), dtype=object)
with open('movies_metadata.csv') as csvFile:
    reader = csv.DictReader(csvFile)
    i = 0
    for row in reader:
        dataEmbeded[i, 0] = row['overview']
        dataEmbeded[i, 1] = row['adult'] == "True"
        dataEmbeded[i, 2] = getFloat(row,'budget')
        dataEmbeded[i, 3] = getNames(row['genres'])
        dataEmbeded[i, 4] = getFloat(row,'popularity')
        dataEmbeded[i, 5] = getNames(row['production_companies'])
        dataEmbeded[i, 6] = row['production_countries'] == "[{'iso_3166_1': 'US', 'name': 'United States of America'}]"
        dataEmbeded[i, 7] = getFloat(row,'revenue')
        dataEmbeded[i, 8] = getNames(row['spoken_languages'])
        i += 1

one_hot = MultiLabelBinarizer()
genres = one_hot.fit_transform(dataEmbeded[:,3])
production_companies = one_hot.fit_transform(dataEmbeded[:,5])
spoken_languages = one_hot.fit_transform(dataEmbeded[:,8])

dataEmbeded = np.delete(dataEmbeded, [3, 5, 8], 1)
dataEmbeded = np.c_[dataEmbeded,genres]
dataEmbeded = np.c_[dataEmbeded,spoken_languages]

## production_companies too large !
#dataEmbeded = np.c_[dataEmbeded,production_companies]

#U,s,V = np.linalg.svd(production_companies, full_matrices=False)
#svdMaker = decomposition.TruncatedSVD(n_components=20, algorithm='arpack', n_iter=10)
#svdMaker.fit(production_companies)

# keep 20 first singular values
#dataEmbeded = np.c_[dataEmbeded, U[:,20]]


# saving
with open('dataEmbeded.pkl', 'wb') as pikeler:
    cPickle.dump(dataEmbeded, pikeler)

""" get the data back
with open('dataEmbeded.pkl', 'rb') as pikeler:
    dataEmbeded = cPickle.load(pikeler)
"""

