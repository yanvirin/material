'''
This scripts runs on datapoints and generates summaries
'''

import os,sys,json
import torch
from torch.autograd import Variable
from collections import namedtuple
sys.path.append("../training")
from utils import write2file
import random
import rouge_papier
from spensum.scripts.baselines.train_rnn_extractor import collect_reference_paths

def get_inputs_metadata(sent_tokens, clean_texts, sen_embds, qry_embds, query="", minTokens = 0):
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
 
  if len(sen_embds) == 0: return None, None

  sen_embds = torch.FloatTensor(sen_embds)
  print("DEBUG: sen embds len: %s" % len(sen_embds))
  qry_embds = torch.FloatTensor([qry_embds]).repeat(len(sen_embds),1)
  embeddings = torch.cat([sen_embds, qry_embds], 1) 
  word_counts = torch.LongTensor([[len(tokens) for tokens in sent_tokens]])
  input_length = torch.LongTensor([len(sent_tokens)])
  inputs = namedtuple("Inputs", ["length", "embedding", "word_count"])(
    Variable(input_length),
    Variable(embeddings.unsqueeze(0)),
    Variable(word_counts.unsqueeze(2)))
  metadata = namedtuple("Metadata", ["text","query"])([clean_texts],[query])
  return inputs, metadata

if __name__ == "__main__":
  
  model_path = sys.argv[1]
  dps_path = sys.argv[2]
  out_path = sys.argv[3]
  refs_path = sys.argv[4]
  rescore = len(sys.argv) > 5 and sys.argv[5] == "rescore"
  strategy = sys.argv[5] if len(sys.argv) > 5 else "rank"
 
  print("predicting with strategy=%s and rescore=%s" % (strategy, rescore))
  os.system("mkdir -p %s" % out_path)

  # load the torch model
  predictor = torch.load(model_path, map_location=lambda storage, loc: storage)
  print("model loaded form %s" % model_path)

  # rouge paths
  rouge_paths = []
  ids2refs = collect_reference_paths(refs_path)

  with rouge_papier.util.TempFileManager() as manager:
   for line in open(dps_path):
     dp = json.loads(line)
     id = dp["id"]
     query = dp["query"]
     #print("dp,id: %s,%s" % (id,query))
     qry_embds = dp["qembedding"]
     sentences = []
     tokens = []
     sen_embds = []
     for input in dp["inputs"]:
       sen_id = input["sentence_id"]
       sen_embds.append(input["embedding"])
       sentences.append(input["text"])
       tokens.append(input["text"].split(" "))
     ref_paths = ids2refs[id]
     inputs, metadata = get_inputs_metadata(tokens, sentences, sen_embds, qry_embds)
     if inputs is not None:
       summaries, _ = predictor.extract(inputs, metadata, strategy=strategy, word_limit=100, rescore=rescore)
       summary_path = "%s/%s.pred" % (out_path, id)
       write2file("%s" % summaries[0] + "\n", summary_path)
       rouge_paths.append([summary_path, ref_paths])

   # compute rouge
   config_text = rouge_papier.util.make_simple_config_text(rouge_paths)
   config_path = manager.create_temp_file(config_text)
   df = rouge_papier.compute_rouge(config_path, max_ngram=2, lcs=False)
   print(df[-1:])
