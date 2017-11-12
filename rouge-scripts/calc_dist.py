import sys, os
from collections import Counter
from collections import defaultdict
from extract_best_sentences import parse_rouge


dist1 = Counter()
dist2 = Counter()
def record(rge, dist):
  v = int(rge * 20) / 20.0
  dist[v] += 1

def printout(dist):
  print "distribution:"
  for k in sorted(dist.keys()):
    print "%f %d" % (k, dist[k])

dir = sys.argv[1]

for d in os.listdir(dir):
  if os.path.isdir(os.path.join(dir, d)):
   mrge1 = 0.0
   mrge2 = 0.0
   for s in os.listdir(os.path.join(dir, d, "sources")):
    if s.endswith(".rge"):
      rge1 = parse_rouge(os.path.join(dir, d, "sources", s), 1)
      rge2 = parse_rouge(os.path.join(dir, d, "sources", s), 2)
      if rge1 > mrge1: mrge1 = rge1
      if rge2 > mrge2: mrge2 = rge2
   record(mrge1, dist1)
   record(mrge2, dist2)

print "total recorded counts: %d" % sum(dist1.values())
printout(dist1)
printout(dist2)
