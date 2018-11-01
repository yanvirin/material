'''
Created on Aug 21, 2018

@author: ezotkina
'''

from Corpus import *


def parse_specs(specs_file):
    fin = open(specs_file, "r")
    lines = fin.readlines()
    fin.close()
    ''' filter_indexes out blank lines'''
    lines = filter(lambda x: not x.isspace(), lines)
    corp = None
    corpora_specs = Corpus_spec()
    location = None
    query = None
    index = None
    is_corpora_specs = False
    is_corpus_specs = False
    is_location = False
    is_query = False
    is_index = False
    
    for line in lines:
        line = line.strip()
        if line.startswith("[corpora_specs]"):
            is_corpora_specs = True
        elif is_corpora_specs == True and not line.startswith("[query_"):
            corpora_specs.add_spec(line)
        
        
        if line.startswith("[query_"):
            is_corpora_specs = False
            is_query = True
            query = Query()
            corpora_specs.add_query(query)            
        elif is_query == True and not line.startswith("[corpus"):
            query.add_spec(line)
        
        
        if line.startswith("[corpus"):
            is_query = False
            is_index = False
            is_corpus_specs = True
            corp = Corpus()
            corp.title = line
            corp.corpus_name = line
            corpora_specs.add_corpus(corp)            
        elif is_corpus_specs == True and not line.startswith("[location]"):
            corp.add_spec(line)
        
        if line.startswith("[location]"):
            is_corpus_specs = False
            is_location = True
            location = Location()
            corp.add_location(location)
        elif is_location == True and not line.startswith("[index]"):
            location.add_spec(line)
        
        
        if line.startswith("[index]"):
            is_index = True
            is_location = False
            index = Index()
            corp.add_index(index)
        elif is_index == True and not line.startswith("[corpus"):
            index.add_spec(line)
        
    return corpora_specs

def get_value(line):
    return line.split("=")[1].strip()
