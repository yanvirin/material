import sys,json


if __name__ == "__main__":
  
  labels_file = sys.argv[1]
  raml_samples_dir = sys.argv[2]
  labels_out_file = sys.argv[3]

  with open(labels_file) as lr:
   with open(labels_out_file, "w") as wr:
    for line in lr:
      d = json.loads(line)
      doc_id = d["id"]
      sub_dir = abs(hash(doc_id)) % 100
      with open("%s/%s/%s.json" % (raml_samples_dir,sub_dir,doc_id)) as r:
        rd = json.loads(r.readlines()[0])
        d["label_scores"] = rd["summaries"]
      wr.write(json.dumps(d)+"\n")
