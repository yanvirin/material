import numpy as np
from collections import defaultdict


class WordEmbeddingDict(object):
    def __init__(self, word_to_index, index_to_word, embeddings, weights=None):
        self.word_to_index_ = word_to_index   
        self.index_to_word_ = index_to_word
        self.embeddings_ = embeddings 
        self.missing_embedding_ = np.array([0.] * embeddings.shape[1])
        self.weights_ = weights

    def __getitem__(self, word_or_index):
        if isinstance(word_or_index, str):
            word = word_or_index
            if word in self.word_to_index_:
                return self.embeddings_[self.word_to_index_[word]]
            else:
                return self.missing_embedding_
        else:
            index = word_or_index
            if index < len(self.index_to_word_) and index >= 0:
                return self.index_to_word_[index]
            else:
                raise Exception("Bad index: {}".format(index))

    def weight(self, token):
        if token in self.word_to_index_:
            return self.weights_[self.word_to_index_[token]]
        else:
            return 1.0

    def sif(self, input_document):
        
        sentence_embeddings = []
        for sentence in input_document:
            if len(sentence) == 0:
                sentence_embeddings.append(self.missing_embedding_)    
            else:
                nonzero = max(
                    sum([t in self.word_to_index_ for t in sentence]), 1)
                emb = np.vstack([self.__getitem__(t) for t in sentence])
                weights = np.array([[self.weight(t) for t in sentence]])
                sent_emb = np.dot(weights, emb) / nonzero
                sentence_embeddings.append(sent_emb)
        return np.vstack(sentence_embeddings)

    def __contains__(self, item):
        return item in self.word_to_index_


def load_word_weights(counts_path, smoothing=1e-3):
    if smoothing <= 0:
        raise Exception("smoothing must be positive")
    total_instances = 0
    word_to_weight = defaultdict(lambda: 1.0)
    with open(counts_path, "r", encoding="utf8") as fp:
        for line in fp:
            word, count = line.strip().split()
            count = float(count)
            word_to_weight[word] = count
            total_instances += count
    for word, count in word_to_weight.items():
        freq = count / total_instances
        word_to_weight[word] = smoothing / (smoothing + freq)

    return word_to_weight 

def load_embeddings(embeddings_path, counts_path=None, smoothing=1e-3):
    word_list = []
    word_map = {}
    embeddings = []
    weights = None
    with open(embeddings_path, "r", encoding="utf8") as fp:
        for i, line in enumerate(fp):
            items = line.strip().split(" ")
            word_list.append(items[0])
            word_map[items[0]] = i
            embeddings.append([float(x) for x in items[1:]])

    if counts_path:
        word_weights = load_word_weights(counts_path, smoothing=smoothing)
        weights = [word_weights[w] for w in word_list]

    return WordEmbeddingDict(word_map, word_list, np.array(embeddings),
                             weights=weights)
