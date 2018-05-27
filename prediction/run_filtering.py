import sys,os
from rnnsum_client import run
sys.path.append("../training")
from utils import write2file
from utils import fileaslist

port = int(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
inputs_path = sys.argv[4]
outputs_dir = sys.argv[5]

os.system("mkdir -p %s" % (outputs_dir))

n = 1000
print("number of shards: %d, start doc %d, end doc %d" % (n,start,end))

data_path = "%s/data%d-%d.txt" % (outputs_dir,start,end)

os.system("> %s" % data_path)
doc = start
d = start / n
if doc != 0: d += 1
with open(data_path, "w") as w:
 while doc <= end:
  if doc % n == 0: d += 1
  os.system("cp %s/%d/%d.query /tmp/yan/query/queries.txt" % (inputs_path,d,doc))
  os.system("cp %s/%d/%d.txt /tmp/yan/inputs/input.txt" % (inputs_path,d,doc))
  run(port)
  summary_sens = fileaslist("/tmp/yan/outputs/input.txt")
  newdoc = " ".join(summary_sens)
  if "\n" in newdoc: raise Exception("new line in the summary!")
  if len(summary_sens) > 0:
    w.write(newdoc + "\n")
  else:
    w.write("\n")
  print("done with document %d" % doc)
  doc += 1
