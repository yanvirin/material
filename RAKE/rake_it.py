import sys,os,rake
sys.path.append("../SIF/examples")
sys.path.append("../SIF/src")
sys.path.append("../training")
import extract_embeddings as em
import utils
import json

'''
Runs the RAKE algorithm on the dataset and created queries and their embeddings
'''

dataset = sys.argv[1]
output_file = sys.argv[2]

# load models for embeddings
wordfile = "../SIF/data/glove.840B.300d-freq500K.txt"
weightfile = "../SIF/auxiliary_data/enwiki_vocab_min200.txt"
(words, We) = em.data_io.getWordmap(wordfile)
word2weight = em.data_io.getWordWeight(weightfile, 1e-3)
weight4ind = em.data_io.getWeight(words, word2weight)

rake_model = rake.Rake("SmartStoplist.txt")

def clean(keywords):
  print "%f - %f" % (keywords[0][1], keywords[1][1])
  return [x[0] for x in keywords]
  #return [x[0] for x in keywords if x[0].split(" ") < 5] + [y[0] for y in keywords]

ids = []
queries = []

for d in os.listdir(dataset):
 if "README" in d: continue
 ids.append(d)
 with open(os.path.join(dataset, d, "content.txt")) as f:
  # replace end of line with a stopword to make the algo break on new lines
  text = f.read() #.replace("\n"," or ")
  keywords = rake_model.run(text)
  if len(keywords) > 0:
    queries.append(clean(keywords)[0])
  else:
    queries.append("")

assert len(ids) == len(queries)

params = em.params.params()
params.rmpc = 0

temp_file = "../../material-data/tmp/queries.txt"
utils.write2file("\n".join(queries), temp_file)

results = []
embeddings = em.get_embeddings(words, We, word2weight, weight4ind, temp_file, params)
for i in range(0, len(queries)):
  results.append({"docset_id": ids[i], "query": queries[i], "embedding": embeddings[i,:].tolist()})

with open(output_file, "w") as w: json.dump(results, w)

