'''
Created on Mar 28, 2018

@author: ezotkina
'''

class Corpus():

    def __init__(self):
        self.specs = []        
        self.locations = []
        self.indexes = []
        #self.name = ""
        #self.corpus_title = ""  
        self.parents = {}
        #self.index_root = ""            
        #self.index_version = ""
        #self.index_params = ""
    
    '''    
    def fill_in_parents(self):
        for location in self.locations:
            self.parents[location.path] = location.get_source()
    '''
        
    def get_parents(self, path):
        parents_list = []
        p = self.parents[path]
          
        while p != None:
            parents_list.insert(0, p)
            p = self.parents[p]
                    
        parents_list.append(path)   
        
        all_parents = ""
        for p in parents_list:
            all_parents += p + "\t"
            
        return all_parents.strip()
        
                
    def add_location(self, location):
        self.locations.append(location)
    
    def add_index(self, index):
        self.indexes.append(index)
    
    def add_spec(self, spec):
        self.specs.append(spec)
        #if spec.lower().startswith("name"):
        #    self.index_root = spec.split("=")[1]
        #if spec.lower().startswith("name"):
        #    self.name = spec.split("=")[1]
            
    def is_text(self):
        for s in self.specs:
            if s.lower().startswith('type'):
                return s.split("=")[1].strip().lower() == 'text'
    
    def get_type(self):
        for s in self.specs:
            if s.lower().startswith('type'):
                return s.split("=")[1].strip().lower()
        
    
    def get_name(self):
        for s in self.specs:
            if s.lower().startswith('name'):
                return s.split("=")[1].strip()
    
    def get_short_name(self):
        '''
        1A/IARPA_MATERIAL_BASE-1A/ANALYSIS1
        it returns ANALYSIS1
        '''
        return self.get_name().split("/")[-1]  
        
    def get_language(self):
        for s in self.specs:
            if s.lower().startswith('language'):
                return s.split("=")[1].strip()
    
    def get_language_id(self):
        all_lang_id = []
        for loc in self.locations:
            lang_id = loc.get_language_id()
            if lang_id:
                all_lang_id.append(self.get_name()+"/"+lang_id)
        
        return all_lang_id
           
    def __repr__(self):
        #return "\nSpecs="+str(self.specs)+";Index_root="+self.index_root+";Index_version="+self.index_version+\
        #    "\nLocations: "+str(self.locations)
        return "\n\nCorpus Specs="+str(self.specs)+\
            "\nLocations: "+str(self.locations)+\
            "\nIndexes: "+str(self.indexes)
    
    def __str__(self):
        return self.__repr__()
    
    
class Corpus_spec():

    def __init__(self):
        self.specs = []
        self.queries = []
        self.corpora = []
    
    def add_spec(self, spec):
        self.specs.append(spec.strip())
    
    def add_query(self, query):
        self.queries.append(query)
    
    def add_corpus(self, corpus):
        self.corpora.append(corpus)
    
    def get_relative_directory(self):
        for s in self.specs:
            if s.lower().startswith('relative_directory'):
                return s.split("=")[1].strip()
    
    def get_spec_version(self):
        for s in self.specs:
            if s.lower().startswith('created'):
                return s.split("=")[1].strip()
    
    def get_languages(self):
        for s in self.specs:
            if s.lower().startswith('all_languages_included'):
                value = s.split("=")[1].strip()
        
        return value.split(";;")
    
    
    def get_corpus_by_name(self, name):
        for corp in self.corpora:
            if corp.get_name() == name:
                return corp
    
    def __repr__(self):
        return "Specs:"+str(self.specs) +\
               "\n\nQueries:" +str(self.queries)+\
               "\n\nCorpora:" +str(self.corpora)
    
    def __str__(self):
        return self.__repr__()
    
    

class Location():
    def __init__(self):
        self.specs = []
        
    def add_spec(self, spec):
        self.specs.append(spec.strip())
    
    def get_language_id(self):        
        for s in self.specs:
            if s.lower().startswith('language_identification_location'):
                return s.split("=")[1].strip()
            
    def __repr__(self):
        return "\nLocation:"+str(self.specs)
    
    
    def __str__(self):
        return self.__repr__()


class Query():
    def __init__(self):
        self.specs = []
        
    def add_spec(self, spec):
        self.specs.append(spec.strip())
    
    def get_language(self):
        for s in self.specs:
            if s.lower().startswith('language'):
                return s.split("=")[1].strip() 
    
    def get_query_location(self):
        for s in self.specs:
            if s.lower().startswith('queryprocessing_location'):
                return s.split("=")[1].strip()
            
    def __repr__(self):
        return "\nQuery:"+str(self.specs)
    
    
    def __str__(self):
        return self.__repr__()
    

class Index():
    def __init__(self):
        self.specs = []
        
    def add_spec(self, spec):
        self.specs.append(spec.strip())
       
    
    def get_location(self):
        for s in self.specs:
            if s.lower().startswith('index_location'):
                return s.split("=")[1].strip() 
    
            
    def __repr__(self):
        return "\nIndex:"+str(self.specs)
    
    
    def __str__(self):
        return self.__repr__()
    
        