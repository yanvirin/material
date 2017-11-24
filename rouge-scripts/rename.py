import sys, os
dir = sys.argv[1]

for d in os.listdir(dir):
  if d != "README":
   for f in os.listdir(os.path.join(dir, d, "sources")):
    if f.endswith(".rouge"):
      os.system("mv %s %s" % (os.path.join(dir, d, "sources", f), os.path.join(dir, d, "sources", f[:-10]+".rge")))

