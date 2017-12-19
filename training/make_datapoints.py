import sys, os, json
sys.path.append("../rouge-scripts")
import make_embeddings as em
import rouge as rge
from multiprocessing.dummy import Pool as threads
import utils
import argparse

'''
This module is responsible to take a duc data input dir and target dir, and
produce a file of datapoints with labels, created greedily using rouge metric with the target

This scripts need theano to be installed on the system.

This script needs to load embeddings and run ROUGE. This combination is a lethal one from
the performance point of view, and so it takes a lot of time to complete.
We want to hold the embeddings models once in RAM,
but want to run ROUGE in parallel. In the future, we need to use more advanced multi processing
libraries in order to share the model as a copy on write memory.
'''

# assumes existance of models variable in the context
def get_embeddings(json_input_file):
  code = abs(hash(json_input_file))
  em.create_embeddings(json_input_file, "%s/mds/%d.norm" % (TMP, code), "%s/mds/%d.emb" % (TMP, code), models)
  return [[float(n) for n in l.split(" ")] for l in utils.fileaslist("%s/mds/%d.emb" % (TMP, code))]

def compute_rouge(sent_list, refs):
  sentext = "\n".join(sent_list)
  code = abs(hash(sentext))
  name0 = "%s/mds/%d.txt" % (TMP, code)
  utils.write2file(sentext, name0)
  cfgline = name0
  for i, sens in enumerate(refs):
    name1 = "%s/mds/%d.txt%d" % (TMP, code, i)
    utils.write2file("\n".join(sens), name1)
    cfgline = cfgline + " " + name1
  cfgfile = "%s/mds/%d.cfg" % (TMP, code)
  utils.write2file(cfgline, cfgfile)
  rouge_out = "%s/mds/%d.rge" % (TMP, code)
  rge.rouge(1000, cfgfile, rouge_out)
  score = rge.parse_rouge(rouge_out, 2) + 0.0001*rge.parse_rouge(rouge_out, 1)
  return score

# needs input_folder and targets_folder defined in the context
def runoninput(f):
  infile = os.path.join(input_folder, f)
  trgfile = ".".join(f.split(".")[:-2] + ["target", "json"])

  # get the ref summaries
  refsums = [d["sentences"] for d in em.read_input(os.path.join(targets_folder, trgfile))]
  
  # compute rouge
  data = zip(em.read_input(infile), get_embeddings(infile))
  
  text = []
  count = 0
  while True:
    changed = False
    for sen, emb in data:
      sen["embedding"] = emb
      sen_text = sen["text"]
      if sen_text not in text:
        sen["label"] = compute_rouge(text + [sen["text"]], refsums)
        changed = True
      else:
        sen["label"] = -1
    if not changed: break
    best = max(data, key=lambda x:x[0]["label"])[0]["text"]
    text.append(best)
    count += len(best.split(" "))
    if count > numwords: break
   
  results = [d[0] for d in data]
  for sen in results: sen["label"] = 1 if sen["text"] in text else 0
  print "processed file %s" % f
  return results

if __name__ == "__main__":
  
  THREADS = 50

  parser = argparse.ArgumentParser(description = 'Create labeled datapoints from the one ducs data folder')
  parser.add_argument('input', metavar="in", help='the path to one input duc data folder')
  parser.add_argument('target', metavar="targ", help='the path to the target corresponding one data folder')
  parser.add_argument('out', metavar="out", help='the path to the output file in json format')
  args = parser.parse_args()

  input_folder = args.input
  targets_folder = args.target
  output_file = args.out
  numwords = 100

  os.chdir("../rouge")

  TMP = "../../material-data/tmp"
  os.system("mkdir -p %s/mds" % TMP)

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
  utils.save_results(results, output_file)
