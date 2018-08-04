import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

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
