import sys,os
from collections import defaultdict

preds_path = sys.argv[1]

dist = defaultdict(int)

for path in os.listdir(preds_path):
  with open("%s/%s" % (preds_path, path)) as f:
    positions = [int(line.split(" ")[0]) for line in f.read().split("\n")]
    for p in positions:
      dist[p] += 1

for i in range(1, max(dist.keys())+1):
  print("%d %d" % (i, dist[i]))
