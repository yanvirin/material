import os, sys
sys.path.append("../rouge-scripts")
import rouge as rge
import utils
import random
import math

rouge_dir = "../rouge"

dataset = sys.argv[1]
ver = sys.argv[2]
eval_out = sys.argv[3]
min_rge = float(sys.argv[4])
typ = sys.argv[5]

FIRST_N_LINES = 3

os.system("mkdir -p /tmp/mds")

eval_path = "/tmp/mds/baseline.cfg"
eval_writer = open(eval_path, "w")

cache = dict()

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

candidate = centroidemd if typ == "centroid" else rand3

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
