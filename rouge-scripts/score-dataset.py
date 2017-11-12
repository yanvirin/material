import sys,os
from os import listdir
from os import path

MAX_WORDS=100
MIN_WORDS=3

dataset = sys.argv[1]
d = sys.argv[2]
content_file = "/tmp/rouge-model%s.txt" % d
config_file = "/tmp/rouge-config%s.txt" % d

if d:
  content = []
  count = 0
  with open(path.join(dataset, d, "content.txt.nrm")) as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      n = len(line.strip().split(" "))
      if n > MIN_WORDS:
        content.append(line)
        count += n
      if n > MAX_WORDS: break

  with open(content_file, "w") as w:
    w.write("\n".join(content))
  
  for s in listdir(path.join(dataset, d, "sources")):
    if s.endswith(".nrm"):
      with open(config_file, "w") as w:
        w.write("%s %s" % (path.join(dataset, d, "sources", s), content_file))
      os.system("./ROUGE-1.5.5.pl -e data -n1 -n2 -l %d -z SPL %s > %s" % (MAX_WORDS, config_file, path.join(dataset, d, "sources", s+".rouge")))
  os.system("rm %s" % content_file)
  os.system("rm %s" % config_file)
