import json,sys,os
import numpy as np

folder = sys.argv[1]
t = 0.0
m = 0
for f in os.listdir(folder):
  data = open("%s/%s" % (folder, f)).readlines()[0]
  d = json.loads(data)
  for (i, ls1) in enumerate(d["label_scores"]):
    if ls1["score"] == 0.0: continue 
    s = 0.0
    n = 0
    for (j, ls2) in enumerate(d["label_scores"]):
      if ls2["score"] == 0.0: continue
      if i != j:
        s += np.dot(np.array(ls1["labels"]), np.array(ls2["labels"]))
        n += 1
    t += s / n
    m += 1
  print("score: %f" % (t/m))
