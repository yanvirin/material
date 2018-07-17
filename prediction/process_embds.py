import os,sys

lines = sys.stdin.readlines()
size = None
for i,line in enumerate(lines):
  line = line.strip().split(" ")
  embd = list(map(lambda x: float(x), line[1:]))
  word = line[0].split(":")[1]
  new_line = " ".join([word] + line[1:]).strip()
  if (len(new_line.split(" ")) == len(line)):
    print(new_line)
