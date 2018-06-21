import os,sys,argparse,time,tempfile,socket,json
import torch
from scipy.spatial import distance as distance
from torch.autograd import Variable
from collections import namedtuple
sys.path.append("../splitta")
sys.path.append("../SIF/examples")
sys.path.append("../SIF/src")
sys.path.append("../training")
import extract_embeddings as em
import sbd
from utils import fileaslist, write2file
import word_tokenize as wt
from predict import get_inputs_metadata
from similarity_extractor import SimilarityExtractor

SUMMARIZATION_TRIGGER = "7XXASDHHCESADDFSGHHSD"

def cossim_weight(u, v):
  raw = min(1.0, max(0.0, 1.0 - distance.cosine(u, v)))
  clamped = raw if raw >= 0.2 else 0.0
  return clamped

def load_stopwords(stopwords_path):
  stopwords = set()
  with open(stopwords_path) as sf:
    for line in sf.readlines():
      stopwords.add(line.strip())
  return stopwords

class Summarizer(object):

    def __init__(self, sif_model, predictor, splitta_model, stopwords):
      self.em = sif_model
      self.predictor = predictor
      self.splitta_model = splitta_model
      self.stopwords = stopwords
    
    def embed_word(self, word):
      if word in self.em[0]: return self.em[1][self.em[0][word]]
      return [100.0]*len(self.em[1][0])  # make sure default is distant from others

    def sum2img(self, summary_dir, query_path):
      # get weights
      query_embd,_ = self.get_query_embd(query_path)
      weights_dir = tempfile.mkdtemp()
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
      os.system("./gen_images.sh %s %s %s" % (summary_dir, weights_dir, summary_dir))
      os.system("rm -r %s" % weights_dir)

    def summarize_text(self, raw_text_path, query, portion=None, max_length=100,rescore=False):
      assert rescore==True or rescore==False
      inputs, metadata = self.ingest_text(raw_text_path, query)
      if inputs == None: return ""
      word_limit = max_length if portion is None else int(inputs.word_count.data.sum()*portion)
      summaries, scores = self.predictor.extract(inputs, metadata, word_limit=word_limit, rescore=rescore)
      return summaries[0]

    '''
    Uses splitta based sentences splitter to 
    split text to sentences and normalize the text
    '''
    def split2sens(self, raw_text_path):
      out_file_name = tempfile.NamedTemporaryFile().name
      with open(out_file_name, "w", encoding="utf-8") as out_f:
        test = sbd.get_data(raw_text_path, tokenize=True)
        test.featurize(self.splitta_model, verbose=False)
        self.splitta_model.classify(test, verbose=False)
        test.segment(use_preds=True, tokenize=False, output=out_f)
      return out_file_name

    def normalize(self, sens_path):
      out_file_name = tempfile.NamedTemporaryFile().name
      write2file("\n".join([wt.normalize(line) for line in fileaslist(sens_path)]), out_file_name)
      return out_file_name
   
    def get_query_embd(self, query_path):
      # extract the query from the query_path
      with open(query_path) as qr:
        query_dict = json.load(qr)
      query = query_dict["parsed_query"][0]["content"]
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

    def ingest_text(self, raw_text_path, query_path):
        sens_text_path = self.split2sens(raw_text_path)
        norm_text_path = self.normalize(sens_text_path)
        sen_embds,qry_embds,query = self.get_embds(norm_text_path, query_path)

        assert(len(fileaslist(sens_text_path)) == len(fileaslist(norm_text_path)))

        clean_texts = fileaslist(sens_text_path)
        sent_tokens = [sen.split(" ") for sen in fileaslist(norm_text_path)]
        return get_inputs_metadata(sent_tokens, clean_texts, sen_embds, qry_embds, query=query)

def get_input_paths(folder, qResults):
  paths = []
  if qResults:
    with open(qResults) as r:
      results = json.load(r)
      for res in results["document info"]["results"]:
        index_toks = res["index"].replace("index_store","mt_store").split("/")
        filename = res["filename"]
        paths.append("%s/%s/%s/%s.txt" % (folder, "/".join(index_toks[:5]), index_toks[-2], filename))
  else:
    for path in os.listdir(folder):
      paths.append("%s/%s" % (folder, path))
  return paths
      

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-folder", default=os.getenv("QUERY_STR"),
        type=str, required=False)
    parser.add_argument(
        "--model-path", default=os.getenv("RNNSUM_PATH"),
        type=str, required=False)
    parser.add_argument(
        "--folder", type=str, required=True)
    parser.add_argument(
        "--results", type=str, required=False, default=None)
    parser.add_argument(
        "--length", default=100, type=int)
    parser.add_argument(
        "--summary-dir", required=True, type=str)
    parser.add_argument(
        "--embd-wordfile-path", required=True, type=str)
    parser.add_argument(
        "--embd-weightfile-path", required=True, type=str)
    parser.add_argument(
        "--port", required=True, type=int)
    parser.add_argument(
        "--rescore", required=True, type=str)
    parser.add_argument(
        "--text-similarity", required=False, type=str, default="False")
    parser.add_argument(
        "--embd-similarity", required=False, type=str, default="False")
    parser.add_argument(
        "--portion", required=False, default=None, type=float)
    parser.add_argument(
        "--stopwords", required=False, default="stopwords.txt", type=str)
    parser.add_argument(
        "--gen-image", required=False, type=str, default="True")
    args = parser.parse_args()
    
    args.rescore=args.rescore=="True"
    args.text_similarity=args.text_similarity=="True"
    args.embd_similarity=args.embd_similarity=="True"
    args.gen_image=args.gen_image=="True"

    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    if args.text_similarity or args.embd_similarity:
      model = SimilarityExtractor(use_text_cosine=args.text_similarity, use_embd_cosine=args.embd_similarity)
      print("Not loading torch models, using similarity.")
    else:
      model = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    
    # initialize the sif embeddings stuff
    (words, We) = em.data_io.getWordmap(args.embd_wordfile_path)
    word2weight = em.data_io.getWordWeight(args.embd_weightfile_path, 1e-3)
    weight4ind = em.data_io.getWeight(words, word2weight)
    embd_params = em.params.params()
    embd_params.rmpc = 0
    sif_model = (words, We, word2weight, weight4ind, embd_params)

    # stopwords
    stopwords = load_stopwords(args.stopwords)    

    # prepare splitta
    splitta_model = sbd.load_sbd_model("../splitta/model_nb/",use_svm=False)

    # create the summarizer
    summarizer = Summarizer(sif_model, model, splitta_model, stopwords)

    # start the server and listen to summarization requests
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(("", args.port))
    serversocket.listen(5)

    print("Loaded all models successfully, ver:06/20/18_12:00PST, ready to accept requests on %d with rescore=%s, portion=%s,similarity text and embd: %s,%s" % (args.port, args.rescore==True, args.portion, args.text_similarity, args.embd_similarity))

    temp_out = tempfile.mkdtemp()
    while 1:
      (clientsocket, address) = serversocket.accept()
      data = clientsocket.recv(1000000)
      params = json.loads(str(data, "utf-8"))
      qExpansion = params["qExpansion"]
      qResults = params["qResults"] if "qResults" in params else None
      input_paths = get_input_paths(args.folder, args.results + "/" + qResults)

      summary_dir = args.summary_dir + "/" + qExpansion
      os.system("mkdir -p %s" % summary_dir)

      # go over all the input files and run summarization
      query_path = os.path.join(args.query_folder,qExpansion)
      for input_path in input_paths:
          summary = summarizer.summarize_text(
                      input_path, query=query_path, portion=args.portion, 
                      max_length=args.length, rescore=args.rescore)
          output_path = os.path.join(temp_out, os.path.basename(input_path))
          with open(output_path, "w", encoding="utf-8") as fp: fp.write(summary)
      if args.gen_image: summarizer.sum2img(temp_out, query_path)
      os.system("mv %s/* %s/" % (temp_out,summary_dir))
      os.system("chmod -R 777 %s" % summary_dir)
      clientsocket.send(SUMMARIZATION_TRIGGER.encode("utf-8"))

if __name__ == "__main__":
    main()
