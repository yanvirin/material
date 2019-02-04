import json,sys

for line in sys.stdin:
  d = json.loads(line)
  print(len(d["inputs"]))


