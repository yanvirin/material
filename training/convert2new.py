import sys,os,json,random

filename = sys.argv[1]
train_portion = float(sys.argv[2])

train_feats_writer =  open(filename+".feats.train.json","w")
valid_feats_writer =  open(filename+".feats.valid.json","w")
train_labels_writer = open(filename+".labels.train.json","w")
valid_labels_writer = open(filename+".labels.valid.json","w")

def writers():
  return (train_feats_writer,train_labels_writer) if random.random() <= train_portion else (valid_feats_writer,valid_labels_writer)

for line in open(filename):
  dl = json.loads(line.strip())
 
  docset_id = None

  inputs = []
  labels = []
  words = 0
  i = 1
  for d in dl:
    assert docset_id == None or docset_id == d["docset_id"]
    docset_id = d["docset_id"]
    word_count = len(d["text"].split(" "))
    words += word_count
    sen_id = int(d["sentence_id"])
    inputs.append({"sentence_id": sen_id, "embedding": d["embedding"], "word_count": word_count, "text": d["text"]})
    labels.append(d["label"])
    
    assert sen_id == i
    i += 1
  
  if words < 100: continue

  feats = dict()
  labs = dict()
  feats["id"] = docset_id
  feats["principal_components"] = [0,0]
  labs["id"] = docset_id
  feats["inputs"] = inputs
  labs["labels"] = labels

  fw,lb = writers() 
  fw.write(json.dumps(feats)+"\n")
  lb.write(json.dumps(labs)+"\n")
