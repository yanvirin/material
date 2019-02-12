import sys,os,re
from nltk.tokenize import word_tokenize

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.system("mkdir -p %s" % output_dir)

m = re.compile("[^A-Za-z0-9]+")

for f in os.listdir(input_dir):
  with open("%s/%s"%(input_dir,f)) as r:
    lines = [l.strip() for l in r.readlines()]
    clean_lines = []
    for line in lines:
      tokens = word_tokenize(line)
      clean_tokens = []
      for token in tokens:
        if m.match(token): continue
        clean_tokens.append(token)
      clean_lines.append(" ".join(clean_tokens))
  with open("%s/%s"%(output_dir,f),"w") as w:
    w.write("\n".join(clean_lines))
      
