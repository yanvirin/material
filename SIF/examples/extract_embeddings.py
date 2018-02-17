from __future__ import print_function
import sys
sys.path.append('../src')
import data_io, params, SIF_embedding
import os

def get_embeddings(words, We, word2weight, weight4ind, filename, params):

  # load sentences
  x, m, _ = data_io.sentiment2idx(filename, words) # x is the array of word indices, m is a mask
  w = data_io.seq2weight(x, m, weight4ind) # get word weights

  # get SIF embedding
  embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
  return embedding

def print_embeddings(embedding, output_file):
  # print out the embeddings
  writer = open(output_file, "w")
  for i in range(0, len(embedding)):
    items = []
    for j in range(0, len(embedding[i, :])):
      items.append(str(embedding[i, j]))
    line = " ".join(items) + "\n"
    writer.write(line)
  writer.close()

if __name__ == "__main__":
  # input
  wordfile = '../data/glove.840B.300d-freq500K.txt' # word vector file, can be downloaded from GloVe website
  weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]
  rmpc = int(sys.argv[3]) # number of principal components to remove in SIF weighting scheme

  params = params.params()
  params.rmpc = rmpc

  # load word vectors
  (words, We) = data_io.getWordmap(wordfile)

  # load word weights
  word2weight = data_io.getWordWeight(weightfile, 1e-3)
  weight4ind = data_io.getWeight(words, word2weight)

  for f in os.listdir(input_dir):
    try:
      input_file = "%s/%s" % (input_dir, f)
      output_file = "%s/%s" % (output_dir, f)
      print_embeddings(get_embeddings(words, We, word2weight, weight4ind, input_file, params), output_file)
    except Exception as ex:
      print("an error occured. skipped %s, due to err: %s" % (input_file, ex))
