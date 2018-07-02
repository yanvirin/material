import os,json,sys
from collections import Counter

d = Counter()
err = 0

for line in sys.stdin:
  l = json.loads(line)
  if len(l) > 0:
    for word in l[0]:
      d[word["word"]] += 1
  


for w,c in d.most_common(10000000):
  print("%s %d" % (w,c))
