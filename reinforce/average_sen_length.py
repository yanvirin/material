import json, os, sys

N = 0
W = 0
D = 0

for f in os.listdir(sys.argv[1]):
  D += 1
  with open("%s/%s" % (sys.argv[1], f)) as r:
    d = json.load(r)
    for input in d["inputs"]:
      N += 1
      W += input["word_count"]

print(W/N)
print(W/D)
    
    
