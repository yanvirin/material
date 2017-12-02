import os,sys,json

sum = 0.0
n = 0.0
for line in sys.stdin:
  sum += len(json.loads(line))
  n += 1

print sum/n
  
