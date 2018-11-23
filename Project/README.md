

```python
import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
import pickle as pkl
from utils.tokenizer import tokenize_corpus
```


```python
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
```


```python
with open('./data/links.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)
    id_to_movieId = dict()
    for line in reader:
        try:
            id_to_movieId[int(line[2])] = int(line[0])
        except:
            pass
```


```python
with open('./data/movies_metadata.csv', encoding= 'utf-8') as csvFile:
    reader = csv.DictReader(csvFile)
    i = 0
    for row in reader:
        dataEmbeded[i, 0] = row['overview']
        try:
            dataEmbeded[i, 1] = id_to_movieId[int(row['id'])]
        except:
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
```


```python
one_hot = MultiLabelBinarizer(sparse_output=True)
genres = one_hot.fit_transform(dataEmbeded[:,4])
production_companies = one_hot.fit_transform(dataEmbeded[:,6])
spoken_languages = one_hot.fit_transform(dataEmbeded[:,9])
BoW = tokenize_corpus(dataEmbeded[:,0], stop_words = False, BoW = True)
```


```python
data =  hstack([BoW, genres, spoken_languages])
with open('./data/data.npy', 'wb') as pikeler:
    data = {'ids':dataEmbeded[:, 1], 'data':data}
    pkl.dump(data, pikeler)
```
