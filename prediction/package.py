import sys,os,uuid,json
from collections import defaultdict
import datetime,argparse,tempfile

SYSTEM_NAME = "system1"
INSTRUCTIONS = "Try to figure out if the output is relevant"

now = datetime.datetime.now().isoformat().split(".")[0] + "Z"

parser = argparse.ArgumentParser()

parser.add_argument("--query-folder", type=str, required=True)
parser.add_argument("--results-folder", type=str, required=True)
parser.add_argument("--summary-dir", type=str, required=True)
parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--package-dir", type=str, required=True)

args = parser.parse_args()

os.system("mkdir -p %s" % args.package_dir)

queries = dict()
data = defaultdict(list)

# write all the data to the files and rename images and summaries
# to fit the correct format
for q_f in os.listdir(args.summary_dir):
  qid = q_f
  res_f = "q-%s.json" % qid
  with open("%s/%s" % (args.query_folder,qid)) as qr: qdict = json.load(qr)
  with open("%s/%s" % (args.results_folder,res_f)) as rr: rdict = json.load(rr)
  with open("%s/%s/s-%s.tsv" % (args.summary_dir,qid,qid),"w") as qw:
    # record the query name and domain
    query_str = "%s:%s" % (qdict["domain"]["desc"],qdict["IARPA_query"])
    qw.write("%s\t%s\n" % (qid,query_str))
    # go over all results and record the needed data
    for res in rdict["document info"]["results"]:
      doc_id = res["filename"]
      relevance = res["score"]
      sum_id = str(uuid.uuid4())
      file_id = "SCRIPTS.%s.%s.%s" % (SYSTEM_NAME,qid,doc_id)
      qw.write("%s\t%f\t%s.json\n" % (doc_id,float(relevance),file_id))
      with open("%s/%s/%s.json" % (args.summary_dir,qid,file_id),"w") as rw:
        os.system("mv %s/%s/%s.png %s/%s/%s.png" % (args.summary_dir,qid,doc_id,args.summary_dir,qid,file_id))
        os.system("mv %s/%s/%s.txt %s/%s/%s.txt" % (args.summary_dir,qid,doc_id,args.summary_dir,qid,file_id))
        with open("%s/%s/%s.txt" % (args.summary_dir,qid,file_id)) as sf:
          word_list = [line.strip().split(" ") for line in sf.readlines()]
          word_list = [item for sublist in word_list for item in sublist]
          assert(len(word_list) <= 100)
        d = dict()
        d["team_id"] = "SCRIPTS"
        d["sys_label"] = SYSTEM_NAME
        d["uuid"] = sum_id
        d["query_id"] = qid
        d["document_id"] = doc_id
        d["run_name"] = args.run_name
        d["run_date_time"] = now
        d["image_uri"] = "%s.png" % file_id
        d["word_list"] = word_list 
        d["instructions"] = INSTRUCTIONS
        json.dump(d,rw)
        rw.write("\n")

# pacakge the stuff
package_path = "%s/summary-package.tgz" % args.package_dir
output_folder = tempfile.mkdtemp()
for q_f in os.listdir(args.summary_dir):
  os.system("tar -czvf %s/%s.tgz -C %s %s" % (output_folder,q_f,args.summary_dir,q_f))
os.system("tar -czvf %s -C %s ." % (package_path,output_folder))
os.system("chmod 777 %s" % package_path)
