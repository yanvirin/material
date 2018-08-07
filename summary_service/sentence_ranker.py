import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import Counter
from sklearn.feature_extraction import DictVectorizer

def query_embedding_similarity(document, query, embeddings):
    query_flat = [term.lower() for subquery in query for term in subquery] 
    query_emb = embeddings.sif([query_flat])

    if np.all(query_emb == 0.):
        logging.warning(" query: {} has no embeddings!".format(query_flat))

    doc_lower = [[t.lower() for t in sent] for sent in document]
    sent_embs = embeddings.sif(doc_lower)
    distance = 1. - cosine_similarity(sent_embs, query_emb).ravel()
    top_indices = np.argsort(distance)
    ranks = np.argsort(top_indices)
    return ranks

def query_lexical_similarity(document, query, query_expansion=None):

    query_dict = Counter([term.lower() for sw in query for term in sw])
    if query_expansion:
        for term, weight in query_expansion:
            if term not in query_dict:
                query_dict[term] = weight
    sent_dicts = [Counter([w.lower() for w in sent]) for sent in document]
    dv = DictVectorizer()
    dv.fit([query_dict] + sent_dicts)

    sent_feats = dv.transform(sent_dicts)
    query_feats = dv.transform(query_dict)

    distance = 1. - cosine_similarity(sent_feats, query_feats).ravel()
    top_indices = np.argsort(distance)
    ranks = np.argsort(top_indices)
    return ranks
