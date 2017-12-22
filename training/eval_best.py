import sys,os
import utils
sys.path.append("../rouge-scripts")
import rouge as rge

dataset = sys.argv[1]
sample = int(sys.argv[2])

for d in os.listdir(dataset)[:sample]:
  ref = utils.fileaslist(os.path.join(dataset, d, "content.txt.nrm"))
  ref = [x for x in ref if len(x.split(" "))>3][:3]
  best_score = 0.0
  best_f = None
  for f in os.listdir(os.path.join(dataset, d, "sources")):
    if f.endswith(".rge"):
      score = rge.parse_rouge(os.path.join(dataset, d, "sources", f), 1)
      if score > best_score:
        best_score = score
        best_f = f 
  best = utils.fileaslist(os.path.join(dataset, d, "sources", best_f[:-3]+"best2"))
  print "=================================================="
  print "\n".join(ref)
  print "--------------------------------------------------"
  print "\n".join(best)
  print "=================================================="

