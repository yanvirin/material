from operator import itemgetter
import string
import numpy as np
import gensim

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from collections import defaultdict


STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add("n't")

def load_topic_model(path):
    if not isinstance(path, str):
        path = str(path)
    return gensim.models.ldamodel.LdaModel.load(path)


def get_topics_helper(model, doc_topics, query, unstemmer, max_output_words):
    if not query in model.id2word.token2id:
        return [], query in unstemmer

    topic_term_matrix = model.get_topics()

    query_idx = model.id2word.token2id[query] 
        
    topic_scores = [np.log(tpc_prob) + \
                        np.log(topic_term_matrix[tpc_idx, query_idx])
                    for tpc_idx, tpc_prob in doc_topics]
    stems_ids = [(stem, model.id2word.token2id[stem]) 
                  for stem in unstemmer.keys()
                  if stem in model.id2word.token2id]
    stems_ids.sort(key=lambda x: x[1])
   
    stems_scores = np.array([float("-inf")] * len(stems_ids))

    for t, topic_lp in enumerate(topic_scores):
        topic_idx = doc_topics[t][0]
        stem_lps = np.log(
            topic_term_matrix[topic_idx][[x[1] for x in stems_ids]])
        stem_lps += topic_lp    
        stems_scores = np.maximum(stems_scores, stem_lps)

    topic_stems = [(stems_ids[s][0], stems_scores[s])
                   for s in np.argsort(stems_scores)[-max_output_words:]]
    return topic_stems, query in unstemmer

def get_topics(model, document, query, max_topic_words,
               always_highlight=False):

    document_preprocessed, unstemmer = preprocess_document(document)
    document_bow = model.id2word.doc2bow(document_preprocessed)
    doc_topics = model.get_document_topics(document_bow)
    query_preprocessed = preprocess_query(query)
    
    word_count = 0
    query_results = []
    for sub_query, sub_query_pp in zip(query, query_preprocessed):

        query_result = {"query": [], "query_highlight": [],
                        "topic_words": []}
        topic_word_scores = defaultdict(lambda: float("-inf"))
        for term, term_stem in zip(sub_query, sub_query_pp):
            
            topic_words, highlight = get_topics_helper(
                model, doc_topics, term_stem, unstemmer, max_topic_words)
            for stem, score in topic_words:
                if score > topic_word_scores[stem]:
                    topic_word_scores[stem] = score
            if always_highlight:
                highlight = True
            query_result["query_highlight"].append(highlight)

        topic_word_scores = sorted(
            topic_word_scores.items(), 
            key=itemgetter(1), 
            reverse=True)
        topic_word_scores = topic_word_scores[:max_topic_words]

        topic_words = []
        for stem, scores in topic_word_scores:
            word = sorted(
                unstemmer[stem].items(), key=itemgetter(1), reverse=True)[0][0]
            topic_words.append(word)
        query_result["topic_words"] = topic_words
        query_result["query"] = sub_query
        query_results.append(query_result)
        word_count += len(topic_words) + len(sub_query)

    return query_results, word_count

def preprocess_document(document):
    output = []
    stemmer = PorterStemmer()
    unstemmer = {}
    for sent in document:
        stemmed_sent = []
        
        for word in sent:
            word = word.lower()
            if word in STOPWORDS or is_punctuation(word):
                continue
            
            stem = stemmer.stem(word)
            if not stem in unstemmer:
                unstemmer[stem] = {}
            if not word in unstemmer[stem]:
                unstemmer[stem][word] = 0
            unstemmer[stem][word] = unstemmer[stem][word] + 1

            stemmed_sent.append(stem)
        output.extend(stemmed_sent)
    return (output, unstemmer)

def is_punctuation(word):
    for char in word:
        if not char in string.punctuation:
            return False
    return True

def preprocess_query(query):
    stemmer = PorterStemmer()    
    preproc_query = []
    for sub_query in query:
        lc_query = [w.lower() for w in sub_query]
        lc_query_stemmed = [stemmer.stem(w) if w not in STOPWORDS else None
                            for w in lc_query]
        preproc_query.append(lc_query_stemmed)
    return preproc_query
