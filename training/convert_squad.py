import os,json,sys,argparse,time,tempfile
sys.path.append("../splitta")
sys.path.append("../SIF/examples")
sys.path.append("../SIF/src")
sys.path.append("../prediction")
import extract_embeddings as em
import sbd
from utils import fileaslist, write2file
import word_tokenize as wt
from rnnsum import Summarizer
import random

parser = argparse.ArgumentParser()

parser.add_argument(
        "--embd-wordfile-path", required=True, type=str)
parser.add_argument(
        "--embd-weightfile-path", required=True, type=str)
parser.add_argument(
        "--squad-data-path", required=True, type=str)
parser.add_argument(
        "--feats-file-path", required=True, type=str)
parser.add_argument(
        "--labels-file-path", required=True, type=str)

args = parser.parse_args(sys.argv[1:])

# load the squad data
data = json.load(open(args.squad_data_path))

# initialize the sif embeddings stuff
(words, We) = em.data_io.getWordmap(args.embd_wordfile_path)
word2weight = em.data_io.getWordWeight(args.embd_weightfile_path, 1e-3)
weight4ind = em.data_io.getWeight(words, word2weight)
embd_params = em.params.params()
embd_params.rmpc = 0
sif_model = (words, We, word2weight, weight4ind, embd_params)

# prepare splitta
splitta_model = sbd.load_sbd_model("../splitta/model_nb/",use_svm=False)

# create the summarizer(we need it only for featurization)
featurizer = Summarizer(sif_model, None, splitta_model)

raw_text_path = tempfile.NamedTemporaryFile().name
query_path = tempfile.NamedTemporaryFile().name

with open(args.feats_file_path,"w") as fw:
 with open(args.labels_file_path,"w") as lw:

  dp_id = 0
  for d in data["data"]:
   dps = []
   # get a squad document
   sens = []
   embeddings = []
   for p in d["paragraphs"]:
     write2file(p["context"],raw_text_path)
     sens_text_path = featurizer.split2sens(raw_text_path)
     norm_text_path = featurizer.normalize(sens_text_path)
     q = random.choice(p["qas"])
     query = q["question"]
     write2file(query,query_path)
     sen_embds,qry_embds,_ = featurizer.get_embds(norm_text_path, query_path)
     dps.append((query,len(sens),len(sen_embds),qry_embds))
     embeddings.extend(sen_embds)
     sens.extend(fileaslist(sens_text_path))
  
   # write dps for the document
   inputs = []
   for i,sen in enumerate(sens):
     inpt = dict()
     inpt["sentence_id"] = i
     inpt["text"] = sen
     inpt["embedding"] = embeddings[i]
     inpt["word_count"] = len(sen.split(" "))
     inputs.append(inpt)
  
   for query,st,cnt,qry_embds in dps:
     dp = dict()
     dp["inputs"] = inputs
     dp["qembedding"] = qry_embds
     dp["query"] = query
     dp["id"] = str(dp_id)
     dp["principal_components"] = [0, 0]
     lb = dict()
     lb["id"] = str(dp_id)
     lb["labels"] = [(1 if i>=st and i<st+cnt else 0) for i in range(len(sens))]
     json.dump(dp,fw)
     json.dump(lb,lw)
     fw.write("\n")
     lw.write("\n")
     dp_id += 1

