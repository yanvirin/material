import sys

words = set()
for line in open(sys.argv[1]):
  words.add(line.strip().split(" ")[0])

sentences = 0.0
coverage = 0.0
for line in sys.stdin:
  ws = line.strip().split(" ")
  if sentences % 10000 == 0: print ws
  n = 0.0
  for w in ws:
    if w in words: n += 1
  coverage += n / len(ws)
  sentences += 1

print coverage / sentences
  
