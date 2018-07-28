# Jessica Ouyang
# lda.py
# Topic modeling


from operator import itemgetter
from os import listdir
import os.path
import string
import sys
import re

import gensim

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


STOPWORDS = stopwords.words('english')        

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

def get_topics_helper(model, query, dictionary, doc, unstemmer, max_output_words):
    output_words = []            
    doc_topics = sorted(model.get_document_topics(dictionary.doc2bow(doc)), key=itemgetter(1), reverse=True)
    topic = None

    stemmer = PorterStemmer()    
    query = stemmer.stem(query)
    if not query in dictionary.token2id:
        topic = doc_topics[0]

    else:
        query_id = dictionary.token2id[query]        
        topics = []
        for topic_id, topic_prob in doc_topics:
            topic_terms = dict([pair for pair in model.get_topic_terms(topic_id, topn=1000)])
            if query_id in topic_terms:
                topics.append((topic_id, topic_terms[query_id]))

        if len(topics) == 0:
            topic = doc_topics[0]
        else:
            topics = sorted(topics, key=itemgetter(1), reverse=True)
            topic = topics[0]
    
    output_counter = 0
    for word_id, word_prob in model.get_topic_terms(topic[0], topn=1000):
        if word_prob < 0.001:
            break

        word = dictionary.id2token[word_id]
        if word in unstemmer:
            output_counter += 1
            word = sorted(unstemmer[word].items(), key=itemgetter(1), reverse=True)[0][0]
            output_words.append((word, word_prob))

            if output_counter == max_output_words:
                break
    return output_words


def get_topics(model, doc, query, max_output_words):
    doc, unstemmer = preprocess2(doc)
    dictionary = model.id2word
    queries = clean_query(query)
    
    all_topic_words = []
    for query in queries:
        topic_words = {}
        for term in query:
            topic = get_topics_helper(model, term, dictionary, doc, unstemmer, max_output_words)
            for word, prob in topic:
                if (not word in topic_words
                            or (word in topic_words and topic_words[word] < prob)):
                    topic_words[word] = prob                    

        topic_words = topic_words.items()
        if len(topic_words) > 0:
            topic_words = sorted(topic_words, key=itemgetter(1), reverse=True)
            if len(topic_words) > max_output_words:
                topic_words = topic_words[:max_output_words]
                
        all_topic_words.append(topic_words)
    return all_topic_words, queries


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
