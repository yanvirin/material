import sys, os, json
sys.path.append("../splitta")
sys.path.append("../SIF/examples")
sys.path.append("../SIF/src")
import extract_embeddings as em 
import word_tokenize as wt

def read_input(fname):
  with open(fname) as f:
    data = json.load(f)
  return data

def create_embeddings(input_fname, norm_fname, embd_fname, models):

  words, We, word2weight, weight4ind = models
  
  wn = open(norm_fname, "w")
  data = read_input(input_fname)
  for dp in data:
    sen = dp["text"]
    wn.write(wt.normalize(sen) + "\n")
    
  wn.close()

  params = em.params.params()
  params.rmpc = 0

  em.print_embeddings(em.get_embeddings(words, We, word2weight, weight4ind, norm_fname, params), embd_fname)

if __name__ == "__main__":

  input_fname = sys.argv[1]
  norm_fname = sys.argv[2]
  embd_fname = sys.argv[3]
  wordfile = sys.argv[4]
  weightfile = sys.argv[5]

  # load word vectors
  (words, We) = em.data_io.getWordmap(wordfile)

  # load word weights
  word2weight = em.data_io.getWordWeight(weightfile, 1e-3)
  weight4ind = em.data_io.getWeight(words, word2weight)

  create_embeddings(input_fname, norm_fname, embd_fname, (words, We, word2weight, weight4ind))
