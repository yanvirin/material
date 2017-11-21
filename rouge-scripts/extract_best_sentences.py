import sys,os
from os import listdir
from os import path

def parse_rouge(f, ver):
  assert ver == 1 or ver == 2
  with open(f) as r:
    lines = r.readlines()
  idx = 1 if ver == 1 else 5
  return float(lines[idx].split(" ")[3])

def rouge(rouge_dir, max_words, config_file, rouge_out):
  os.system("%s/ROUGE-1.5.5.pl -e data -n1 -n2 -l %d -z SPL %s > %s" % (rouge_dir, max_words, config_file, rouge_out))

if __name__ == "__main__":

  MAX_WORDS=100
  MIN_WORDS=3

  dataset = sys.argv[1]
  ver = int(sys.argv[2])
  d = sys.argv[3]
  
  content_file = "/tmp/rouge-model%s.txt" % d
  config_file = "/tmp/rouge-config%s.txt" % d
  source_sen_file = "/tmp/rouge-source-sen%s.txt" % d
  rouge_out = "/tmp/rouge-out%s.txt" % d

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
        if n > MAX_WORDS: break

    with open(content_file, "w") as w:
      w.write("\n".join(content))
  
    for s in listdir(path.join(dataset, d, "sources")):
      if s.endswith(".nrm"):
        with open(path.join(dataset, d, "sources", s)) as f: 
          sentences = map(lambda x: x.strip(), f.readlines())
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
        with open(path.join(dataset, d, "sources", s[:-4]+".best"+str(ver)), "w") as w:
          for ss in best: w.write(ss+"\n")
    os.system("rm %s" % content_file)
    os.system("rm %s" % config_file)
    os.system("rm %s" % source_sen_file)
    os.system("rm %s" % rouge_out)
