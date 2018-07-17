import json,os,sys
from collections import defaultdict

out_dir = sys.argv[1]

os.system("mkdir -p %s/results" % out_dir)
os.system("mkdir -p %s/query" % out_dir)
os.system("rm -f %s/results/*" % out_dir)
os.system("rm -f %s/query/*" % out_dir)

qd = dict()
rd = defaultdict(list)

for line in sys.stdin.readlines()[1:]:
  doc_id,_,query_id,query_string,_,document_path,_ = line.strip().split("\t")
  index = "/".join(document_path.split("/")[4:]).replace("mt_store","index_store")
  if query_id not in qd:
    qd[query_id] = { "parsed_query": [ { "content": query_string } ] }
  rd[query_id].append({"index":index,"filename":doc_id})

for query_id in qd:
  query_file = "%s/query/%s" % (out_dir,query_id)
  results_file = "%s/results/%s.json" % (out_dir,query_id)
  with open(query_file,"w") as qw: json.dump(qd[query_id],qw)
  resd = {"document info": { "results": rd[query_id] } }
  with open(results_file,"w") as rw: json.dump(resd,rw)
