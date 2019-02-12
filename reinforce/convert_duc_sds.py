import sys, json, os
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
from multiprocessing import Pool

def create_dp(processing_input):

  dp_id, orig_dir, json_datapoint_dir = processing_input
  
  print("Processing %s from %s with outputs into %s" % processing_input)
  
  orig_path = "%s/%s" % (orig_dir, dp_id)
  with open(orig_path) as r:
    content = [l.strip() for l in r.readlines()]
  
  # write the json datapoint
  json_datapoint_path = "%s/%s.json" % (json_datapoint_dir, dp_id)
  dp = dict()
  dp["id"] = dp_id
  inputs = []
  dp["inputs"] = inputs
  i = 0
  for line in content:
    for sen in sent_tokenize(line):
      i += 1
      input = dict()
      input["tokens"] = word_tokenize(sen)
      input["text"] = " ".join(input["tokens"])
      input["sentence_id"] = i
      input["word_count"] = len(input["tokens"])
      inputs.append(input)
  
  with open(json_datapoint_path, "w") as w:
    json.dump(dp, w)

if __name__ == "__main__":
  
  JOBS = 12
  pool = Pool(JOBS)
   
  orig_dir = sys.argv[1]
  datapoints_dir = sys.argv[2]

  os.system("mkdir -p %s" % datapoints_dir)

  files = os.listdir(orig_dir)
  processing_inputs = []
  for f in files:
    dp_id = f.upper()
    processing_inputs.append((dp_id,orig_dir,datapoints_dir))
    if len(processing_inputs) == JOBS:
      pool.map(create_dp, processing_inputs)
      processing_inputs = []
  if len(processing_inputs) > 0:
    pool.map(create_dp, processing_inputs)
