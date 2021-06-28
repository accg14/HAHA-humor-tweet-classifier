import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab


def load(root_model):
    with open(root_model + '.txt',encoding='utf-8') as f:
        index_vocab = [t.strip() for t in f]
    vocab = {w:Vocab(index=i) for (i,w) in enumerate(index_vocab)}
    vectors = np.load(root_model + '.npy')
    m = Word2Vec()
    m.wv.index2word = index_vocab
    m.wv.vectors = vectors

    m.wv.vocab = vocab
    return m
