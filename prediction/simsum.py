import os,sys,argparse,time,tempfile,socket,json,traceback
from pathlib import Path
import re
import numpy as np
import torch
from scipy.spatial import distance as distance
from torch.autograd import Variable
from collections import namedtuple
sys.path.append("../splitta")
sys.path.append("../SIF/examples")
sys.path.append("../SIF/src")
sys.path.append("../training")
sys.path.append("../lda")
import extract_embeddings as em
import sbd
import lda
from utils import fileaslist, write2file
import word_tokenize as wt
from predict import get_inputs_metadata
from similarity_extractor import SimilarityExtractor
from borda_count_rank_merge import borda_count_rank_merge

SUMMARIZATION_TRIGGER = "7XXASDHHCESADDFSGHHSD"

def cossim_weight(u, v):
  raw = min(1.0, max(0.0, 1.0 - distance.cosine(u, v)))
  clamped = raw if raw >= 0.2 else 0.0
  return clamped

def get_translated_query(query_data):
    trans_data = query_data["translations"][3]
    assert trans_data["Indri_query"].startswith("#combine(") \
        and trans_data["Indri_query"].endswith(")")
    indri_str = trans_data["Indri_query"][9:-1]

    results = []
    for item in re.findall(r"(\w+)|#wsyn\((.*?)\)", indri_str):
        if item[0] == '':
            # found a weighted synset of (prob word) pairs.
            prob_words = item[1].split(" ")
            result = []
            for i in range(0, len(prob_words), 2):
                prob, word = prob_words[i:i+2]
                prob = float(prob)
                result.append((prob, word))
            results.append(result)
        else:
            # There was no translation -- use english query word and hope!
            results.append([(1.0, item[0])])
    out_query = " ".join([x[0][1] for x in results])
    if DEBUG: print("translated query: %s" % out_query)
    return out_query

def load_stopwords(stopwords_path):
  stopwords = set()
  with open(stopwords_path,encoding="utf-8") as sf:
    for line in sf.readlines():
      stopwords.add(line.strip())
  return stopwords

class Summarizer(object):

    def __init__(self, sif_model, predictor, stopwords, translate_query, language):
      self.em = sif_model
      self.predictor = predictor
      self.stopwords = stopwords
      self.translate_query = translate_query
      self.language = language
    
    def embed_word(self, word):
      if word in self.em[0]: return self.em[1][self.em[0][word]]
      return [100.0]*len(self.em[1][0])  # make sure default is distant from others

    def sum2img(self, summary_dir, query_path, highlight):
      # get weights
      query_embd,_ = self.get_query_embd(query_path)
      weights_dir = tempfile.mkdtemp()
      try:
       for summary_fn in os.listdir(summary_dir):
         weights = []
         summary_path = os.path.join(summary_dir, summary_fn)
         for sen in fileaslist(summary_path):
           sen_weights = []
           for word in sen.split(" "):
             word_embd = self.embed_word(word)
             weight = 0.0 if word.lower() in self.stopwords else cossim_weight(word_embd, query_embd)
             assert(weight >= 0.0 and weight <= 1.0)
             sen_weights.append(weight)
           weights.append([str(w) for w in sen_weights])
         write2file("\n".join([" ".join([str(w) for w in ws]) for ws in weights]),os.path.join(weights_dir, summary_fn))
      
       # gen image
       os.system("./gen_images.sh %s %s %s %s" % (summary_dir, weights_dir, summary_dir, highlight))
      finally: 
       os.system("rm -r %s" % weights_dir)

    def summarize_text(self, raw_text_path, out_text_path, query, portion=None, max_length=100,rescore=False):
      assert rescore==True or rescore==False
      inputs, metadata = self.ingest_text(raw_text_path, out_text_path, query)
      if inputs == None: return ""
      word_limit = max_length if portion is None else int(inputs.word_count.data.sum()*portion)
      summaries, indices = self.predictor.extract(inputs, metadata, word_limit=word_limit, rescore=rescore)
      return summaries[0], indices[0]

    '''
    Remove empty lines from the raw input
    '''
    def split2sens(self, raw_text_path):
        out_file_name = tempfile.NamedTemporaryFile().name
        with open(out_file_name,"w",encoding="utf-8") as w:
          for line in fileaslist(raw_text_path):
            line = line.strip()
            if len(line) > 0: w.write(line + "\n")
        return out_file_name

    def normalize(self, sens_path):
      out_file_name = tempfile.NamedTemporaryFile().name
      write2file("\n".join([wt.normalize(line) for line in fileaslist(sens_path)]), out_file_name)
      return out_file_name
  
    def get_query_embd(self, query_path):
      
      query = get_query(query_path, self.translate_query)
      
      # deal with query
      qin_f = tempfile.NamedTemporaryFile()
      write2file(wt.normalize(query), qin_f.name)
      qout_f = tempfile.NamedTemporaryFile()
      em.print_embeddings(em.get_embeddings(self.em[0],self.em[1],self.em[2],self.em[3],qin_f.name,self.em[4]), qout_f.name)
      qry_embds = [float(x) for x in fileaslist(qout_f.name)[0].split(" ")]
      return qry_embds,query

    def get_embds(self, norm_text_path, query_path):
   
      # deal with the text
      out_f = tempfile.NamedTemporaryFile()
      em.print_embeddings(em.get_embeddings(self.em[0],self.em[1],self.em[2],self.em[3],norm_text_path,self.em[4]), out_f.name) 
      sen_embds = [[float(x) for x in line.split(" ")] for line in fileaslist(out_f.name)]     
      qry_embds,query = self.get_query_embd(query_path)
      return sen_embds, qry_embds, query

    def ingest_text(self, raw_text_path, out_text_path, query_path):
        sens_text_path = self.split2sens(raw_text_path)
        sens_text_path2 = self.split2sens(out_text_path)
        norm_text_path = self.normalize(sens_text_path)
        sen_embds,qry_embds,query = self.get_embds(norm_text_path, query_path)

        assert(len(fileaslist(sens_text_path)) == len(fileaslist(norm_text_path)))
        if DEBUG: print("compare sizes: %d - %d" % (len(fileaslist(sens_text_path2)),len(fileaslist(norm_text_path))))
        assert(len(fileaslist(sens_text_path2)) == len(fileaslist(norm_text_path)))

        clean_texts = fileaslist(sens_text_path2)
        sent_tokens = [sen.split(" ") for sen in fileaslist(norm_text_path)]
        return get_inputs_metadata(sent_tokens, clean_texts, sen_embds, qry_embds, query=query)

def get_query(query_path, translate_query):
      # extract the query from the query_path
      with open(query_path,encoding="utf-8") as qr:
        query_dict = json.load(qr)
      query = query_dict["parsed_query"][0]["content"] if not translate_query else get_translated_query(query_dict)
      return query

# decides how to get the input texts
def get_input_paths(folder, qResults, language):
  paths = []
  if os.path.isfile(qResults):
    with open(qResults,encoding="utf-8") as r:
      results = json.load(r)
      for res in results["document info"]["results"]:
        index = res["index"]
        index_toks = index.replace("index_store","mt_store").split("/")
        filename = res["filename"]
        ep = "%s/%s/%s/%s.txt" % (folder, "/".join(index_toks[:5]), index_toks[-2], filename)
        if language == "en":
          paths.append((ep,ep))
        else:
          # check that the correct laguage was selected in the server
          assert(language=="sw" and "1A/" in index or language=="tl" and "1B/" in index)
          input_name = tempfile.NamedTemporaryFile().name
          index_toks = index.replace("index_store","morphology_store").split("/")
          morpho_store = "%s/%s" % (folder, "/".join(index_toks[:5]))
          if DEBUG: print("looking in morpho store: %s" % morpho_store)
          morpho_ver = list(filter(lambda x: "morph-v3.0" in x.name and ("v5.0" in x.name or "audio" not in ep), sorted(Path(morpho_store).iterdir(), key=lambda f: f.stat().st_mtime)))[-1].name
          #list(filter(lambda x: "morph-v3.0" in x, os.listdir(morpho_store)))[0]
          input_file = "%s/%s/%s.txt" % (morpho_store, morpho_ver, filename)
          with open(input_name, "w",encoding="utf-8") as w:
           with open(input_file,encoding="utf-8") as r:
            for line in r:
              d = json.loads(line)
              if len(d) > 0: 
                w.write(" ".join(map(lambda x: x["word"], d[0])) + "\n")
              else:
                w.write("empty.\n")
          paths.append((input_name, ep))
          if (len(fileaslist(input_name))!=len(list(filter(lambda x: len(x)>0,fileaslist(ep))))): 
            if DEBUG: print("DEBUG: diff sizes %s vs %s" % (input_file, ep))
  else:
    for path in os.listdir(folder):
      p = "%s/%s" % (folder, path)
      paths.append((p,p))
  return paths

def get_summarizer(args, language, stopwords, translate_query):
      
      embds_path = "%s/%s" % (args.embds_dir,("en_%s_embds.txt"%language))
      weights_path = "%s/%s" % (args.embds_dir,("en_%s_freq.txt"%language))

      # initialize the sif embeddings stuff
      (words, We) = em.data_io.getWordmap(embds_path)
      embd_dim = len(We[0])
      word2weight = em.data_io.getWordWeight(weights_path, 1e-3)
      weight4ind = em.data_io.getWeight(words, word2weight)
      embd_params = em.params.params()
      embd_params.rmpc = 0
      sif_model = (words, We, word2weight, weight4ind, embd_params)

      predictor = SimilarityExtractor(use_text_cosine=args.text_similarity, use_embd_cosine=args.embd_similarity,
              embd_dim=embd_dim)
      return Summarizer(sif_model=sif_model, predictor=predictor, stopwords=stopwords, translate_query=translate_query, language=language)

def load_summarizers(args):

    # stopwords
    stopwords = load_stopwords(args.stopwords)
    summarizers = []

    for sum_typ in args.configuration.split("|"):

      if sum_typ == "en":
        summarizer = get_summarizer(args, "en", stopwords, False)
        summarizers.append({"tl":summarizer,"sw":summarizer})
      if sum_typ == "cross":
        tl_summarizer = get_summarizer(args, "tl", stopwords, False)
        sw_summarizer = get_summarizer(args, "sw", stopwords, False)
        summarizers.append({"tl":tl_summarizer,"sw":sw_summarizer})
      if sum_typ == "src":
        tl_summarizer = get_summarizer(args, "tl", stopwords, True)
        sw_summarizer = get_summarizer(args, "sw", stopwords, True)
        summarizers.append({"tl":tl_summarizer,"sw":sw_summarizer})

    return summarizers

def order_sens(summary, indices):
  sens = summary.split("\n")
  in_order = [None]*len(sens)
  for i,j in enumerate(indices):
    in_order[j] = sens[i]
  return in_order

def get_additional_content(source_path, query_path):
  if TOPIC:
    query = get_query(query_path, translate_query=False)
    topics,queries = lda.get_topics(TOPIC, fileaslist(source_path), query, 5)
    queries = list(map(lambda x: " ".join(x), queries))
    if DEBUG: print("topics: %s, queries: %s" % (topics,queries))
    if len(topics) == 0: return "",0
    assert(len(topics)==len(queries))
    content = "; ".join(list(map(lambda x: "%s: %s" % (x[0],", ".join(list(map(lambda y: y[0], x[1])))), zip(queries, topics))))
  else:
    return "",0

  return content, len(content.split(" "))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-folder", type=str, required=True)
    parser.add_argument(
        "--folder", type=str, required=True)
    parser.add_argument(
        "--results", type=str, required=False, default="")
    parser.add_argument(
        "--length", default=100, type=int)
    parser.add_argument(
        "--summary-dir", required=True, type=str)
    parser.add_argument(
        "--embds-dir", required=True, type=str)
    parser.add_argument(
        "--port", required=True, type=int)
    parser.add_argument(
        "--text-similarity", required=False, type=str, default="False")
    parser.add_argument(
        "--embd-similarity", required=False, type=str, default="True")
    parser.add_argument(
        "--portion", required=False, default=None, type=float)
    parser.add_argument(
        "--stopwords", required=False, default="stopwords.txt", type=str)
    parser.add_argument(
        "--gen-image", required=False, type=str, default="True")
    parser.add_argument(
        "--workDir", required=False, type=str, default=".")
    parser.add_argument(
        "--highlight", required=False, type=str, default="None")
    parser.add_argument(
        "--translate-query", required=False, type=str, default="False")
    parser.add_argument(
        "--configuration", required=False, type=str, default="en|cross|src")
    parser.add_argument(
        "--topic-model-path", required=False, type=str, default="")
    parser.add_argument(
        "--debug", required=False, type=str, default="False")
    args = parser.parse_args()
    
    args.text_similarity=args.text_similarity=="True"
    args.embd_similarity=args.embd_similarity=="True"
    args.gen_image=args.gen_image=="True"

    global DEBUG
    DEBUG=args.debug=="True"

    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    global TOPIC
    if os.path.exists(args.topic_model_path):
      TOPIC=lda.load_topic_model(args.topic_model_path)
    else:
      TOPIC=None

    summarizers = load_summarizers(args)

    print("Loaded server on port %s, with %d summarizers and topic model: %s" % (args.port, len(summarizers), TOPIC))

    # start the server and listen to summarization requests
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(("", args.port))
    serversocket.listen(5)

    temp_out = tempfile.mkdtemp()
    while 1:
      (clientsocket, address) = serversocket.accept()
      data = clientsocket.recv(1000000)
      params = json.loads(str(data, "utf-8"))
      qExpansion = params["qExpansion"]
      qResults = params["qResults"] if "qResults" in params else "None"
      summary_dir = args.summary_dir + "/" + qExpansion
      query_path = os.path.join(args.query_folder,qExpansion)
      os.system("mkdir -p %s" % summary_dir)

      # get inputs, no matter which summarizer to use to the langugage of the summarizer
      input_paths = list(map(lambda s: get_input_paths(args.folder, args.results + "/" + qResults, s["tl"].language),
                         summarizers))
      try:
        # go over all the input files and run summarization for all summarizers
        for i in range(len(input_paths[0])):
          try:
            # topic model content
            add_content, add_size = get_additional_content(input_paths[0][i][1], query_path=query_path)
            length = 0
            final_summary_sens = []
            rankings = []
            sens_in_order = None
            for s in range(len(summarizers)):
              if DEBUG: print("DEBUG: working on %s and %s" % (input_paths[s][i][0], input_paths[s][i][1]))
              key = "sw" if "1A/" in input_paths[s][i][1] else "tl"
              summary, indices = summarizers[s][key].summarize_text(input_paths[s][i][0], input_paths[s][i][1],
                               query=query_path, portion=args.portion, max_length=sys.maxsize, rescore=False)
              if not sens_in_order: sens_in_order = order_sens(summary, indices)
              rankings.append(np.argsort(indices))
            max_length = args.length - add_size
            final_indices = np.argsort(borda_count_rank_merge(rankings))
            for idx in final_indices:
              if length >= max_length: break
              candidate = sens_in_order[idx]
              words = candidate.split(" ")
              if length + len(words) <= max_length:
                final_summary_sens.append(candidate)
                length += len(words)
              else:
                final_summary_sens.append(" ".join(words[:max_length-length]))
                length = max_length
            final_summary = "\n".join(final_summary_sens) if not add_content else "\n".join([add_content] + final_summary_sens)
            output_path = os.path.join(temp_out, os.path.basename(input_paths[0][i][1]))
            with open(output_path, "w", encoding="utf-8") as fp: fp.write(final_summary)
          except: traceback.print_exc()
        if args.gen_image: summarizers[0]["tl"].sum2img(temp_out, query_path, args.highlight)
        os.system("mv %s/* %s/ 2> /dev/null" % (temp_out,summary_dir))
        os.system("chmod -R 777 %s" % summary_dir)
      except: traceback.print_exc()
      clientsocket.send(SUMMARIZATION_TRIGGER.encode("utf-8"))

if __name__ == "__main__":
    main()
