import os, sys, json
from multiprocessing.dummy import Pool as threads
import utils
sys.path.append("../rouge-scripts")
from rouge import parse_rouge
import argparse

'''
This script converts a dataset to a labeled dataset
in the json format.
'''

MDS_TYPE = "mds"
SDS_TYPE = "sds"

def runformds(datapoint_folder):
  dsf = dataset_folder
  dpf = datapoint_folder
  srf = os.path.join(dsf, dpf, "sources")
  docset_id = datapoint_folder
  max_rouge = -1.0
  best_source = None
  for source in os.listdir(srf):
    if source.endswith(".rge"): 
      score = parse_rouge(os.path.join(srf, source), 1)
      if score > max_rouge:
        max_rouge = score
        best_source = ".".join(source.split(".")[:-1])
  if not best_source: return []
  best = set(utils.fileaslist(os.path.join(srf, "%s.best%d" % (best_source, ver))))
  return labeled(srf, orig_best = best, orig_docset_id = docset_id)

def runforsds(datapoint_folder):
  srf = os.path.join(dataset_folder, datapoint_folder, "sources")
  return labeled(srf)

def labeled(srf, orig_best = None, orig_docset_id = None):
  output = []
  n = 0
  curr = []
  last_docset_id = None
  for source in os.listdir(srf):
    base = ".".join(source.split(".")[:-1])
    embedding = os.path.join(srf, base+".emd")
    if source.endswith(".nrm") and os.path.exists(embedding):
      doc_id = source.split(".")[0]
      sentences = utils.fileaslist(os.path.join(srf, source))
      embeddings = [[float(y) for y in x.split(" ")] for x in utils.fileaslist(embedding)]
      if parse_rouge(os.path.join(srf, base+".rge"), 2) < min_rge: continue
      best = orig_best if orig_best else set(utils.fileaslist(os.path.join(srf, "%s.best%d" % (base, ver))))
      docset_id = orig_docset_id if orig_docset_id else doc_id
      if docset_id != last_docset_id:
        if len(curr) > 0: output.append(curr)
        curr = []
      last_docset_id = docset_id
      for i, sen in enumerate(sentences):
        d = dict()
        n += 1
        d["docset_id"] = docset_id
        d["doc_id"] = doc_id
        d["sentence_id"] = n if orig_best else str(i+1) 
        d["embedding"] = embeddings[i]
        d["label"] = 1 if sen in best else 0
        d["text"] = sentences[i]
        if len(d) > 0: curr.append(d)
  if len(curr) > 0: output.append(curr)  
  return output

def runoninput(datapoint_folder):
  mds = typ == MDS_TYPE
  if mds:
    return runformds(datapoint_folder)
  else:
    return runforsds(datapoint_folder)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description = 'Convert a dataset to a labeled dataset for training or evaluation')
  parser.add_argument('dataset', metavar="ds", help='the path to the dataset top level folder')
  parser.add_argument('outfile', metavar="out", help='the output file with the labedled data points as a json file')
  parser.add_argument('ver', metavar="ver", type=int, help='whether to use .best1 or .best2 to create the datapoints')
  parser.add_argument('min_rge', metavar="mrge", type=float, help='the minimal ROUGE2 in the .rge file to consider as valid')
  parser.add_argument('typ', help='to create mds or sds datapoints (mds, sds)')
  args = parser.parse_args()
  
  dataset_folder = args.dataset
  output_file = args.outfile
  ver = args.ver
  min_rge = args.min_rge
  typ = args.typ
 
  THREADS = 20

  # go over the dataset and fill results with json dictionaries
  pool = threads(THREADS)
  output = pool.map(runoninput, filter(lambda x: "README" not in x, os.listdir(dataset_folder)))
  results = [item for sublist in output for item in sublist] 
  utils.save_results(results, output_file)
