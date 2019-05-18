import sys,os
import numpy as np
from functools import cmp_to_key

prefixes = ["FT","SJMN","WSJ","AP","FBIS","LA"]

input_dir = sys.argv[1]
output_dir = sys.argv[2]

def compare(x, y):
  def score(name):
    for i,prefix in enumerate(prefixes):
      if name.startswith(prefix):
        return i
    return len(prefixes)
  return score(x) - score(y)

os.system("mkdir -p %s" % output_dir)

for d in os.listdir(input_dir):
  files = sorted(os.listdir("%s/%s"%(input_dir,d)), key=cmp_to_key(compare))
  sens = []
  for f in files:
    with open("%s/%s/%s"%(input_dir,d,f)) as r:
      sens.extend([l.strip() for l in r.readlines()])
  with open("%s/%s"%(output_dir,d.upper()),"w") as w:
    w.write("\n".join(sens))
