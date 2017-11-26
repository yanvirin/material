import os, sys, json
from multiprocessing.dummy import Pool as threads
import utils
sys.path.append("../rouge-scripts")
from rouge import parse_rouge

MDS_TYPE = "mds"
SDS_TYPE = "sds"

class Options(object):
  pass

def runformds(datapoint_folder):
  dsf = options.dataset_folder
  dpf = datapoint_folder
  ver = options.ver
  srf = os.path.join(dsf, dpf, "sources")
  docset_id = datapoint_folder
  max_rouge = -1.0
  best_source = None
  for source in os.listdir(srf):
    if source.endswith(".rge"): 
      score = parse_rouge(os.path.join(srf, source), options.ver)
      if score > max_rouge:
        max_rouge = score
        best_source = ".".join(source.split(".")[:-1])
  if not best_source: return []
  best = set(utils.fileaslist(os.path.join(srf, "%s.best%d" % (best_source,ver))))
  return labeled(srf, orig_best = best, orig_docset_id = docset_id)

def runforsds(datapoint_folder):
  dsf = options.dataset_folder
  dpf = datapoint_folder
  srf = os.path.join(dsf, dpf, "sources")
  return labeled(srf)

def labeled(srf, orig_best = None, orig_docset_id = None):
  ver = options.ver
  min_rge = options.min_rge
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
      best = orig_best if orig_best else set(utils.fileaslist(os.path.join(srf, "%s.best%d" % (base,ver))))
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
  mds = options.typ == MDS_TYPE
  if mds:
    return runformds(datapoint_folder)
  else:
    return runforsds(datapoint_folder)

if __name__ == "__main__":

  options = Options()
 
  dataset_folder = sys.argv[1]
  output_file = sys.argv[2]
  ver = int(sys.argv[3]) # rouge 1 or 2
  min_rge = float(sys.argv[4]) # minimum rouge score
  typ = sys.argv[5] # mds or sds

  options.dataset_folder = dataset_folder
  options.ver = ver
  options.typ = typ
  options.min_rge = min_rge
 
  THREADS = 20

  # go over the dataset and fill results with json dictionaries
  pool = threads(THREADS)
  output = pool.map(runoninput, filter(lambda x: "README" not in x, os.listdir(dataset_folder)))
  results = [item for sublist in output for item in sublist] 
  utils.save_results(results, output_file)
