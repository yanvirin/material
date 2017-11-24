import os

def parse_rouge(f, ver):
  assert ver == 1 or ver == 2
  with open(f) as r:
    lines = r.readlines()
  idx = 1 if ver == 1 else 5
  return float(lines[idx].split(" ")[3])

# has to be called from within rouge dir
def rouge(max_words, config_file, rouge_out):
  os.system("./ROUGE-1.5.5.pl -e data -n 1 -n 2 -n 4 -m -a -x -c 95 -r 1000 -f A -p 0.5 -t 0 -l %d -z SPL %s > %s" % (max_words, config_file, rouge_out))
