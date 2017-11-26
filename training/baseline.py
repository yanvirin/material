import os, sys
sys.path.append("../rouge-scripts")
import rouge as rge
import utils

rouge_dir = "../rouge"

dataset = sys.argv[1]
ver = sys.argv[2]
eval_out = sys.argv[3]
min_rge = float(sys.argv[4])

FIRST_N_LINES = 3

os.system("mkdir -p /tmp/mds")

eval_path = "/tmp/mds/baseline.cfg"
eval_writer = open(eval_path, "w")

for dp in os.listdir(dataset):
  if dp == "README": continue
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
    if os.path.exists(os.path.join(sources, s)) and os.path.exists(text_path):
      if rge.parse_rouge(os.path.join(sources, base+".rge"), 2) < min_rge: continue
      can_text = "\n".join(utils.fileaslist(text_path)[:FIRST_N_LINES])
      can_path = "/tmp/mds/%s.can.txt" % base
      utils.write2file(can_text, can_path)
      eval_writer.write("%s %s\n" % (os.path.join(sources, s), can_path))

eval_writer.close()
print "created the evaluation file, running rouge..."

os.chdir(rouge_dir)
rge.rouge(1000, eval_path, eval_out)

print "done."
