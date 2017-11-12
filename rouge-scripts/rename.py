import sys, os
dir = sys.argv[1]

for d in os.listdir(dir):
  for f in os.listdir(os.path.join(dir, d, "sources")):
    if f.endswith(".best"):
      os.system("mv %s %s" % (os.path.join(dir, d, "sources", f), os.path.join(dir, d, "sources", f[:-5]+".best1")))

