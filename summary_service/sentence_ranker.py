import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from allennlp.predictors.predictor import Predictor
import math

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

def query_qa_similarity(document, query, question_word):
    
    predictor = Predictor.from_path(
                "https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    
    question = " ".join(question_word.split("-")) + " is " + " and ".join(
               [" ".join(sub_query) for sub_query in query]) + " ?"
    print("question: %s" % question)

    # get tokens from each sentence
    document = [predictor.predict(passage=" ".join(sent), 
                        question=question)["passage_tokens"] for sent in document]

    # ask the question
    passage = "\n".join([" ".join(sent) for sent in document])
    answer = predictor.predict(passage=passage, question=question)

    distances = []
    st_idx = 0
    end_idx = 0
    for sent in document:
      end_idx = st_idx + len(sent)
      distances.append(1. - (math.log(max(answer["span_start_probs"][st_idx:end_idx])) + math.log(
                                   max(answer["span_end_probs"][st_idx:end_idx]))))
      st_idx = end_idx
    top_indices = np.argsort(distances)
    ranks = np.argsort(top_indices)
    return ranks
