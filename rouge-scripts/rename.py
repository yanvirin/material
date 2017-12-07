import sys, os
dir = sys.argv[1]

for d in os.listdir(dir):
  if "README" not in d:
   for f in os.listdir(os.path.join(dir, d, "sources")):
    if f.endswith(".best_seq"):
      os.system("mv %s %s" % (os.path.join(dir, d, "sources", f), os.path.join(dir, d, "sources", f[:-9]+".best2")))

