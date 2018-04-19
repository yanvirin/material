import sys,os
from rnnsum_client import run

port = int(sys.argv[1])
analysis_inputs_path = sys.argv[2]
analysis_outputs_dir = sys.argv[3]

os.system("mkdir -p %s" % (analysis_outputs_dir))

for line in open(analysis_inputs_path).readlines()[1:]:
  line = line.strip()
  values = line.split("\t")
  doc_id = values[0]
  query_id = values[2]
  query = values[3]
  doc_path = values[5]

  print("doing: %s %s %s %s" % (doc_id,query_id,query,doc_path))
  os.system("cp %s %s" % (doc_path, "/tmp/yan/inputs/input.txt"))
  with open("/tmp/yan/query/queries.txt", "w") as wq: wq.write(query)
  run(port)
  os.system("mv %s %s/%s.%s.summary" % ("/tmp/yan/outputs/input.txt",analysis_outputs_dir,query_id,doc_id))
