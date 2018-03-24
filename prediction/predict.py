'''
This scripts runs on datapoints and generates summaries
'''

import os,sys,json
import torch
from torch.autograd import Variable
from collections import namedtuple
sys.path.append("../training")
from utils import write2file

def get_inputs_metadata(sent_tokens, clean_texts, sen_embds, qry_embds, minTokens = 3):
  # filter out very short sentences
  short = []
  for i,tokens in enumerate(sent_tokens): 
    if len(tokens) < minTokens: short.append(i)
  for i in sorted(short, reverse=True):
    del sent_tokens[i]
    del clean_texts[i]
    del sen_embds[i]
  assert(len(sent_tokens) == len(clean_texts))
  assert(len(clean_texts) == len(sen_embds))
 
  sen_embds = torch.FloatTensor(sen_embds)
  qry_embds = torch.FloatTensor([qry_embds]).repeat(len(sen_embds),1)
  embeddings = torch.cat([sen_embds, qry_embds], 1) 
  word_counts = torch.LongTensor([[len(tokens) for tokens in sent_tokens]])
  input_length = torch.LongTensor([len(sent_tokens)])
  inputs = namedtuple("Inputs", ["length", "embedding", "word_count"])(
    Variable(input_length),
    Variable(embeddings.unsqueeze(0)),
    Variable(word_counts.unsqueeze(2)))
  metadata = namedtuple("Metadata", ["text"])([clean_texts])
  return inputs, metadata

if __name__ == "__main__":
  
  model_path = sys.argv[1]
  dps_path = sys.argv[2]
  out_path = sys.argv[3]
  rescore = len(sys.argv) > 4 and sys.argv[4] == "rescore"
 
  print("predicting with rescore=%s" % rescore) 
  os.system("mkdir -p %s" % out_path)

  # load the torch model
  predictor = torch.load(model_path, map_location=lambda storage, loc: storage)
  print("model loaded form %s" % model_path)

  for line in open(dps_path):
    dp = json.loads(line)
    id = dp["id"]
    query = dp["query"]
    qry_embds = dp["qembedding"]
    sentences = []
    tokens = []
    sen_embds = []
    for input in dp["inputs"]:
      sen_id = input["sentence_id"]
      sen_embds.append(input["embedding"])
      sentences.append("%s %s" % (str(sen_id), input["text"]))
      tokens.append(input["text"].split(" "))
    inputs, metadata = get_inputs_metadata(tokens, sentences, sen_embds, qry_embds)
    summary = predictor.extract(inputs, metadata, word_limit=100, rescore=rescore)[0]
    write2file("query: " + query + "\n" + summary + "\n", "%s/%s.pred" % (out_path, id))
