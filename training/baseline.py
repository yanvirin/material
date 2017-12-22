import argparse
import os, sys
sys.path.append("../rouge-scripts")
import rouge as rge
import utils
import random
import math
from collections import defaultdict

'''
This script runs the baseline evaluation on the dataset
It can run the CentroidEmd, First3 (Lead Three) or random baselines
The output is a ROUGE script output which includes ROUGE1 and ROUGE2 numbers
'''

FIRST_N_LINES = 3

def kl(source_path):
  def dist(text):
    words = text.split(" ")
    d = defaultdict(int)
    for w in words: d[w] += 1.0 / len(words)
    return d

  sentences = utils.fileaslist(source_path)
  D_dist = dist(" ".join(sentences))
  best = []
  for j in range(FIRST_N_LINES):
    min_dist = 100000
    best_sen = None
    for s in sentences:
      if s in best: pass
      candidate = best + [s]
      S_dist = dist(" ".join(candidate))
      distance = 0.0
      for w in s.split(" "): distance += -S_dist[w] * math.log(D_dist[w]/(S_dist[w]+0.00000000001), 2.0)
      if distance < min_dist:
        min_dist = distance
        best_sen = s
    if best_sen: 
      best.append(best_sen)
  return "\n".join(best)

def centroidemd(source_path):
  get_embds = lambda path: [[float(y) for y in x.split(" ")] for x in utils.fileaslist(path[:-3]+"emd")]
  if source_path not in cache:
    source_embds = get_embds(source_path)
    cache[source_path] = utils.average(source_embds)

  source_sens = utils.fileaslist(source_path)
  centroid = cache[source_path]

  assert len(source_sens) == len(source_embds)

  best = set()
  for j in range(FIRST_N_LINES):
    try:
      best.add(max(set(range(len(source_embds)))-best, key=lambda i: utils.cosine_similarity(source_embds[i],centroid)))
    except ValueError: print "too small text"

  return "\n".join([source_sens[i] for i in best])

def rand3(text_path):
  l = utils.fileaslist(text_path)
  return "\n".join(random.sample(l,min(FIRST_N_LINES,len(l))))

def first3(text_path):
  return "\n".join(utils.fileaslist(text_path)[:FIRST_N_LINES])


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description = 'Compute baselines')
  parser.add_argument('dataset', metavar="ds", help='the path to the dataset top level folder')
  parser.add_argument('ver', type=int, help='ROUGE ver to user either 1 or 2 for choosing the best source to use (mds)')
  parser.add_argument('eval_out', metavar="out", help='the path of the output ROUGE file; should be global path')
  parser.add_argument('min_rge', metavar="mrge", type=float, help='the minimal ROUGE2 in the .rge file to consider as valid')
  parser.add_argument('typ', help='the type of baseline to run: centroid, first3, rand3')
  args = parser.parse_args()  

  dataset = args.dataset
  ver = str(args.ver)
  eval_out = args.eval_out
  min_rge = args.min_rge
  typ = args.typ

  rouge_dir = "../rouge"

  os.system("mkdir -p /tmp/mds")

  eval_path = "/tmp/mds/baseline.cfg"
  eval_writer = open(eval_path, "w")

  cache = dict()

  candidate = centroidemd if typ == "centroid" else (first3 if typ == "first3" else kl if typ == "kl" else rand3)

  print "using %s for candidates" % candidate

  for dp in os.listdir(dataset):
    if dp == ".README": continue
    sources = os.path.join(dataset, dp, "sources")
    best = None
    max_score = 0.0
    for s in os.listdir(sources):
      if s.endswith(".rge"):
        score = rge.parse_rouge(os.path.join(sources, s), int(ver))
        if score > max_score:
          max_score = score
          best = s
    if best:
      base = ".".join(best.split(".")[:-1])
      s = base+".best"+ver
      text_path = os.path.join(sources, base+".nrm")
      cont_path = os.path.join(dataset, dp, "content.txt.nrm")
      if os.path.exists(os.path.join(sources, s)) and os.path.exists(text_path):
        if rge.parse_rouge(os.path.join(sources, base+".rge"), 2) < min_rge: continue
        can_text = candidate(text_path)
        ref_text = "\n".join([x for x in utils.fileaslist(cont_path) if len(x.split(" "))>3][:FIRST_N_LINES])
        can_path = "/tmp/mds/%s.can.txt" % base
        ref_path = "/tmp/mds/%s.ref.txt" % base
        utils.write2file(can_text, can_path)
        utils.write2file(ref_text, ref_path)
        eval_writer.write("%s %s\n" % (can_path, ref_path))

  eval_writer.close()
  print "created the evaluation file, running rouge..."

  os.chdir(rouge_dir)
  rge.rouge(1000, eval_path, eval_out)

  print "done."
