import sys,os
sys.path.append("../training")
from utils import fileaslist

d1 = sys.argv[1]
d2 = sys.argv[2]

s = 0.0
n = 0
for f1 in os.listdir(d1):
  pred1 = set(fileaslist("%s/%s" % (d1,f1)))
  pred2 = set(fileaslist("%s/%s" % (d2,f1)))
  score = float(len(pred1.intersection(pred2))) / len(pred1.union(pred2))
  s += score
  n += 1

print("avg similarity: %f" % (s / n))
  
