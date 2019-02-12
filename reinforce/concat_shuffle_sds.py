import sys,os
import numpy as np

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.system("mkdir -p %s" % output_dir)

np.random.seed(1000)

for d in os.listdir(input_dir):
  sens = []
  for f in os.listdir("%s/%s"%(input_dir,d)):
    with open("%s/%s/%s"%(input_dir,d,f)) as r:
      sens.extend([l.strip() for l in r.readlines()])
  np.random.shuffle(sens)
  with open("%s/%s"%(output_dir,d.upper()),"w") as w:
    w.write("\n".join(sens)) 
