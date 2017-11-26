import json

def fileaslist(f):
  with open(f) as fh: return map(lambda x: x.decode("utf-8").strip(), fh.readlines())

def write2file(text, f):
  with open(f, "w") as fw: fw.write(text.encode('utf-8'))

def sort_results(results):
  for r in results:
    r.sort(cmp=lambda x,y: cmp(x["docset_id"],y["docset_id"])*100+cmp(x["doc_id"],y["doc_id"])*10+cmp(int(x["sentence_id"]),int(y["sentence_id"])))
  results.sort(cmp=lambda x,y: cmp(x[0]["docset_id"],y[0]["docset_id"]))

def save_results(results, output_file):
  sort_results(results)

  # write the results out
  with open(output_file, 'w') as outfile:
    for i,r in enumerate(results):
      json.dump(r, outfile)
      if i != len(results)-1: outfile.write("\n")
