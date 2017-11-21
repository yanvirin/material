import sys, os, json
import make_embeddings as em
sys.path.append("../rouge-scripts")
import extract_best_sentences as rge
from multiprocessing.dummy import Pool as threads
import utils

'''
This module is responsible to take a duc data input dir and target dir, and
produce a file of datapoints with labels, created greedily using rouge metric with the target
'''

# assumes existance of models variable in the context
def get_embeddings(json_input_file):
  code = abs(hash(json_input_file))
  em.create_embeddings(json_input_file, "/tmp/mds/%d.norm" % code, "/tmp/mds/%d.emb" % code, models)
  return [[float(n) for n in l.split(" ")] for l in utils.fileaslist("/tmp/mds/%d.emb" % code)]

def quantize(data, numwords):
  sorted_sc_data = sorted(data, key=lambda x: -x["label"])
  lbl = 1
  wordcount = 0
  for sen in sorted_sc_data:
    sen["label"] = lbl
    wordcount += len(em.wt.normalize(sen["text"]).split(" "))
    if wordcount > numwords:
      lbl = 0

  return sorted_sc_data

def compute_rouge(sentext, refs, ver = 1):
  code = abs(hash(sentext))
  name0 = "/tmp/mds/%d.txt" % code
  utils.write2file(sentext, name0)
  cfgline = name0
  for i, sens in enumerate(refs):
    name1 = "/tmp/mds/%d.txt%d" % (code, i)
    utils.write2file("\n".join(sens), name1)
    cfgline = cfgline + " " + name1
  cfgfile = "/tmp/mds/%d.cfg" % code
  utils.write2file(cfgline, cfgfile)
  rouge_out = "/tmp/mds/%d.rge" % code
  rge.rouge(".", 1000, cfgfile, rouge_out)
  return rge.parse_rouge(rouge_out, ver)

# needs input_folder and targets_folder defined in the context
def runoninput(f):
  print "processing file %s" % f
  infile = os.path.join(input_folder, f)
  trgfile = ".".join(f.split(".")[:-2] + ["target", "json"])

  # get the ref summaries
  refsums = [d["sentences"] for d in em.read_input(os.path.join(targets_folder, trgfile))]
  
  # compute rouge
  data = zip(em.read_input(infile), get_embeddings(infile))
  
  for sen, emb in data:
    sen["embedding"] = emb
    sen["label"] = compute_rouge(sen["text"], refsums)

  # quantize the label scores
  return quantize([x[0] for x in data], numwords)

if __name__ == "__main__":
  
  THREADS = 20

  input_folder = sys.argv[1]
  targets_folder = sys.argv[2]
  output_file = sys.argv[3]
  numwords = int(sys.argv[4])

  os.system("mkdir -p /tmp/mds")
  os.chdir("../rouge")

  # load models for embeddings
  wordfile = "../SIF/data/glove.840B.300d-freq500K.txt"
  weightfile = "../SIF/auxiliary_data/enwiki_vocab_min200.txt"
  (words, We) = em.em.data_io.getWordmap(wordfile)
  word2weight = em.em.data_io.getWordWeight(weightfile, 1e-3)
  weight4ind = em.em.data_io.getWeight(words, word2weight)
  models = (words, We, word2weight, weight4ind) 

  # a list of dictionaries, each representing
  # a sentence from some document and some docset

  pool = threads(THREADS)
  results = pool.map(runoninput, os.listdir(input_folder))
  results = [item for sb in results for item in sb]
  utils.sort_results(results)
 
  # write the results out
  with open(output_file, 'w') as outfile: json.dump(results, outfile)
