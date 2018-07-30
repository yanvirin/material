# Jessica Ouyang
# lda.py
# Topic modeling


from operator import itemgetter
from os import listdir
import os.path
import string
import sys
import re
import numpy as np

import gensim

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from collections import defaultdict

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add("n't")

def get_topic_model(doc_dir):
    docs = []
    for doc_name in listdir(doc_dir):
        doc = None
        with open(os.path.join(doc_dir, doc_name)) as docf:
            doc = [line.strip() for line in docf]
        doc = preprocess1(doc)
        if doc:
            docs.append(doc)

    dictionary = gensim.corpora.Dictionary(docs)
    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(doc) for doc in docs]    
    return gensim.models.ldamodel.LdaModel(corpus, num_topics=10000, id2word=dictionary)

def save_topic_model(model, filename):
    model.save(filename)
def load_topic_model(filename):
    return gensim.models.ldamodel.LdaModel.load(filename)

def is_punctuation(word):
    for char in word:
        if not char in string.punctuation:
            return False
    return True


def preprocess1(doc):
    try:
        doc = [word_tokenize(sent) for sent in doc]
    except UnicodeDecodeError:
        return None

    output = []
    stemmer = PorterStemmer()
    for sent in doc:        
        stemmed_sent = []
        
        for word in sent:
            word = word.lower()
            #try:
            #    word = word.encode('ascii', errors='ignore')
            #except UnicodeDecodeError:
            #    continue
            if word in STOPWORDS or is_punctuation(word):
                continue
            
            stem = stemmer.stem(word)
            stemmed_sent.append(stem)
        output.extend(stemmed_sent)
    return output
        
def preprocess2(doc):
    doc = [word_tokenize(sent) for sent in doc]

    output = []
    stemmer = PorterStemmer()
    unstemmer = {}
    for sent in doc:
        stemmed_sent = []
        
        for word in sent:
            word = word.lower()
            #try:
            #    word = word.encode('ascii', errors='ignore')
            #except UnicodeDecodeError:
            #    continue
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


def clean_query_helper(query):
    query = re.sub(r"\[(hyp|evf|syn):(.*?)\]", r" \2 ", query)
    index = query.find('[')
    if index > 0:
        query = query[:index]
    if '<' in query:
        query = query.replace('<', '').replace('>', '')
    if '"' in query:
        query = query.replace('"', '')

    query = query.replace("+", " ")
    query = re.sub(r"EXAMPLE_OF\((.*?)\)", r"\1", query)

    return [word.lower() for word in query.split() 
            if not word.lower() in STOPWORDS]

def clean_query(query):
    query = query.split(',')
    return [clean_query_helper(subquery) for subquery in query]

def get_topics_helper(doc_topics, topic_term_matrix, query, dictionary, 
                      unstemmer, max_output_words):
    stemmer = PorterStemmer()    
    query = stemmer.stem(query)
    if not query in dictionary.token2id:
        return [], query in unstemmer
    query_idx = dictionary.token2id[query] 
        
    topic_scores = [np.log(tpc_prob) + \
                        np.log(topic_term_matrix[tpc_idx, query_idx])
                    for tpc_idx, tpc_prob in doc_topics]
    stems_ids = [(stem, dictionary.token2id[stem]) 
                  for stem in unstemmer.keys()
                  if stem in dictionary.token2id]
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

def get_topics(model, doc, query, max_output_words):
    doc, unstemmer = preprocess2(doc)
    dictionary = model.id2word
    queries = clean_query(query)

    doc_topics = model.get_document_topics(dictionary.doc2bow(doc))
    topic_term_matrix = model.get_topics()

    word_count = 0
    query_results = []
    for query in queries:
        query_result = {"query": [], "query_highlight": [],
                        "topic_words": []}
        topic_word_scores = defaultdict(lambda: float("-inf"))
        for term in query:
            
            topic_words, highlight = get_topics_helper(
                doc_topics, topic_term_matrix, term, dictionary, 
                unstemmer, max_output_words)
            for stem, score in topic_words:
                if score > topic_word_scores[stem]:
                    topic_word_scores[stem] = score
            query_result["query_highlight"].append(highlight)

        topic_word_scores = sorted(
            topic_word_scores.items(), 
            key=itemgetter(1), 
            reverse=True)
        topic_word_scores = topic_word_scores[:max_output_words]

        topic_words = []
        for stem, scores in topic_word_scores:
            word = sorted(
                unstemmer[stem].items(), key=itemgetter(1), reverse=True)[0][0]
            topic_words.append(word)
        query_result["topic_words"] = topic_words
        query_result["query"] = query
        query_results.append(query_result)
        word_count += len(topic_words) + len(query)

    return query_results, word_count

def main(args):
    #doc_dir = args[0]
    #model_filename = args[1]

    #model = get_topic_model(doc_dir)
    #save_topic_model(model, model_filename)

    manifest_filename = args[0]
    doc_dir = args[1]
    output_dir = args[2]
    model_filename = args[3]
    max_output_words = int(args[4])

    model = load_topic_model(model_filename)
    
    manifest = None
    with open(manifest_filename) as inf:
        manifest = [line.strip().split('\t') for line in inf]        
            
    for line in manifest[1:]:
        doc_name = line[0]        
        query_id = line[2]
        query = line[3]

        doc = None
        with open(os.path.join(doc_dir, doc_name)) as docf:
            doc = [line.strip() for line in docf]   
        
        output = []
        topics = get_topics(model, doc, query, max_output_words)
        for topic in topics:
            output.append((query, topic))
                
        '''with open(os.path.join(output_dir, '%s.%s.topics' % (query_id, doc_name)), 'w') as outf:
            for query, topic in output:
                print >>outf, 'QUERY: %s' % query
                for pair in topic:
                    print >>outf, '%s\t%f' % pair'''
                        
        

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
