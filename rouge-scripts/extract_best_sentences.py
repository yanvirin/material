import sys,os
from os import listdir
from os import path
from rouge import rouge
from rouge import parse_rouge

'''
The version of choosing the best greedily and independently
of the previous choises, meaning each sentence is evaluated with rouge
against the summary reference independenly of any other sentences.
'''
def choosebest_ind(sentences):
  scores = []
  for i in range(0, len(sentences)):
    with open(source_sen_file, "w") as w: w.write(sentences[i])
    rouge(MAX_WORDS, config_file, rouge_out)
    score = parse_rouge(rouge_out, ver)
    scores.append(score)
  count = 0
  best = []
  for sen, scr in sorted(zip(sentences, scores), key=lambda(x,y): -y):
    best.append(sen)
    count += len(sen.split(" "))
    if count > MAX_WORDS: break
  return best

def choosebest_seq(sentences):
  best = []
  count = 0
  while True:
    mscore = -1
    msen = None
    for sen in sentences:
      if sen in best: continue
      with open(source_sen_file, "w") as w: w.write("\n".join(best + [sen]))
      rouge(MAX_WORDS, config_file, rouge_out)
      score = parse_rouge(rouge_out, ver)
      if score > mscore:
        mscore = score
        msen = sen
    if msen is None: break
    best.append(msen)
    count += len(msen.split(" "))
    if count > MAX_WORDS: break
  return best

if __name__ == "__main__":

  MAX_WORDS=100
  MIN_WORDS=3

  tmpdir="../../material-data/tmp"

  os.system("mkdir -p %s" % tmpdir)  

  dataset = sys.argv[1]
  ver = int(sys.argv[2]) # rouge 1 or 2
  d = sys.argv[3]
  bestver = sys.argv[4] # choosing best with "ind" or "seq"
 
  content_file = "%s/rouge-model%s.txt" % (tmpdir, d)
  config_file = "%s/rouge-config%s.txt" % (tmpdir, d)
  source_sen_file = "%s/rouge-source-sen%s.txt" % (tmpdir, d)
  rouge_out = "%s/rouge-out%s.txt" % (tmpdir, d)

  with open(config_file, "w") as w:
    w.write("%s %s" % (source_sen_file, content_file))

  with open(source_sen_file, "w") as w: w.write("")
  with open(rouge_out, "w") as w: w.write("") 

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
        if count > MAX_WORDS: break

    with open(content_file, "w") as w:
      w.write("\n".join(content))
  
    for s in listdir(path.join(dataset, d, "sources")):
      if s.endswith(".nrm"):
        with open(path.join(dataset, d, "sources", s)) as f: 
          sentences = map(lambda x: x.strip(), f.readlines())
        best = choosebest_ind(sentences) if bestver == "ind" else choosebest_seq(sentences)
        with open(path.join(dataset, d, "sources", s[:-4]+".best_"+str(bestver)), "w") as w:
          for ss in best: w.write(ss+"\n")
    os.system("rm %s" % content_file)
    os.system("rm %s" % config_file)
    os.system("rm %s" % source_sen_file)
    os.system("rm %s" % rouge_out)
