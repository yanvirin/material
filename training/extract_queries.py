import sys,os,json

queries_path = sys.argv[1]
dataset_path = sys.argv[2]
feats_path = sys.argv[3]
valid_path = sys.argv[4]

def extract_query_embeds():
 embeds = dict()
 # form query embeddings
 for d in json.loads(open(queries_path).read()):
  query = d["query"]
  docset_id = d["docset_id"]
  embedding = d["embedding"]
  for f in os.listdir("%s/%s/sources" % (dataset_path,docset_id)):
    if f.endswith(".txt"):
      document_id = f.split(".")[0]
      embeds[document_id] = (query,embedding)
 return embeds

def enchance_feats_with_embeds(path, embeds):
  # enchance the features file with query embeddings
  with open(path + ".query","w") as w:
   for line in open(path):
    dp = json.loads(line)
    document_id = dp["id"]
    dp["query"] = embeds[document_id][0]
    dp["qembedding"] = embeds[document_id][1]
    w.write(json.dumps(dp)+"\n")

query_embeds = extract_query_embeds()

print "loaded embeds"

enchance_feats_with_embeds(feats_path, query_embeds)
enchance_feats_with_embeds(valid_path, query_embeds)

print "done."
