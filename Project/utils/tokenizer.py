import spacy
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
import itertools
from collections import Counter
cores = cpu_count()

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

def tokenize_with_stopwords(text):
    return [token.text for token in nlp(text)]

def tokenize_without_stopwords(text):
    return [token.text for token in nlp(text) if not token.is_stop]

def get_vocab(tokenized_corpus):
    words = list(itertools.chain.from_iterable(tokenized_corpus))
    return Counter(words).most_common()

def index_corpus(tokenized_corpus, vocab):
    vocab_indexer = dict([(word[0], i) for i, word in enumerate(vocab)])
    indexed_corpus = []
    for sentence in tokenized_corpus:
        indexed_corpus.append([vocab_indexer[word] for word in sentence])
    return indexed_corpus

def to_BoW(indexed_corpus, vocabulary):
    indptr = [0]
    indices = []
    data = []
    for d in indexed_corpus:
        for index in d:
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), dtype=int)

def tokenize_corpus(corpus, stop_words=True, BoW=True):
    with Pool(processes=cores) as p:
        tokenized_corpus = p.map(tokenize_with_stopwords, corpus) if stop_words else p.map(tokenize_without_stopwords, corpus)
    vocab = get_vocab(tokenized_corpus)
    indexed_corpus = index_corpus(tokenized_corpus, vocab)
    if BoW:
        output = to_BoW(indexed_corpus, vocab)
    else:
        output = indexed_corpus
    return output

