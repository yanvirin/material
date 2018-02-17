import sys,os
sys.path.append("../rouge-scripts")
import rouge as rge

dataset = sys.argv[1]

for d in os.listdir(dataset):
  if d == ".README": continue
  best = None
  best_score = None
  for f in os.listdir("%s/%s/sources" % (dataset, d)):
    if ".rge" in f:
       score = rge.parse_rouge("%s/%s/sources/%s" % (dataset, d, f), 1)
       if best_score is None or best_score < score:
         best_score = score
         best = f.split(".")[0]
  print best
