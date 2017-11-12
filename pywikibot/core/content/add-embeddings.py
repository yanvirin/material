import os, sys

sen_file = sys.argv[1]
emb_file = sys.argv[2]
dataset_dir = sys.argv[3]

embeddings = dict()
vectors = []

for line in open(emb_file, "r"):
  numbers = line.strip().split(" ")
  floats = map(float, numbers)
  vectors.append(floats)

i = 0
for line in open(sen_file, "r"):
  embeddings[line.strip()] = vectors[i]
  i += 1

print "Loaded vectors."

def get_embeddings(sentences):
  results = []
  for s in sentences:
    results.append(embeddings[s.strip()])
  return results

def write_embds(path):
  with open(path) as fd:
    content = fd.readlines()
    embds = get_embeddings(content)
  with open(".".join(path.split(".")[:-1]) + ".emd", "w") as fd:
    for e in embds:
      l = " ".join(map(lambda n: str(n), e)) + "\n"
      fd.write(l)

for d in os.listdir(dataset_dir):
  write_embds("%s/%s/content.txt.nrm" % (dataset_dir, d))
  for s in os.listdir("%s/%s/sources" % (dataset_dir, d)):
    if s.endswith(".nrm"):
      write_embds("%s/%s/sources/%s" % (dataset_dir, d, s))
    
