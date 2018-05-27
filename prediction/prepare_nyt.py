import sys,os

sys.path.append("../RAKE")
sys.path.append("../training")
import rake
from utils import write2file

max_docs = 1000

data_path = sys.argv[1]
output_path = sys.argv[2]

rake_model = rake.Rake("../RAKE/SmartStoplist.txt")

d = 0
doc = 0
for line in open(data_path):
  if doc % max_docs == 0:
    d += 1
    os.system("mkdir -p %s/%d" % (output_path,d))
  text = line.strip()
  res = rake_model.run(text)
  query = res[0][0]
  text = "\n".join(text.split(" . "))
  write2file(text, "%s/%d/%d.txt" % (output_path,d,doc))
  write2file(query, "%s/%d/%d.query" % (output_path,d,doc))
  doc += 1
  
