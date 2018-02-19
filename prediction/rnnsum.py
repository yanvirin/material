import os,sys,argparse,time,tempfile,socket
import torch
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

SUMMARIZATION_TRIGGER = "7XXASDHHCESADDFSGHHSD"

class Summarizer(object):

    def __init__(self, sif_model, predictor, splitta_model):
      self.em = sif_model
      self.predictor = predictor
      self.splitta_model = splitta_model

    def summarize_text(self, raw_text_path, query, max_length=100):
      inputs, metadata = self.ingest_text(raw_text_path, query)
      summary = self.predictor.extract(inputs, metadata, word_limit=max_length)[0]
      return summary

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
   
    def get_embds(self, norm_text_path, query_path):

      # extract the query from the query_path
      query = fileaslist(query_path)[0]
      
      # deal with the text
      out_f = tempfile.NamedTemporaryFile()
      em.print_embeddings(em.get_embeddings(self.em[0],self.em[1],self.em[2],self.em[3],norm_text_path,self.em[4]), out_f.name) 
      sen_embds = torch.FloatTensor([[float(x) for x in line.split(" ")] for line in fileaslist(out_f.name)])
      
      # deal with query
      qin_f = tempfile.NamedTemporaryFile()
      write2file(wt.normalize(query), qin_f.name)
      qout_f = tempfile.NamedTemporaryFile()
      em.print_embeddings(em.get_embeddings(self.em[0],self.em[1],self.em[2],self.em[3],qin_f.name,self.em[4]), qout_f.name)
      qry_embds = torch.FloatTensor([[float(x) for x in fileaslist(qout_f.name)[0].split(" ")]]).repeat(len(sen_embds),1)
      return torch.cat([sen_embds, qry_embds], 1)

    def ingest_text(self, raw_text_path, query_path):
        sens_text_path = self.split2sens(raw_text_path)
        norm_text_path = self.normalize(sens_text_path)
        sentence_embeddings = self.get_embds(norm_text_path, query_path)

        assert(len(fileaslist(sens_text_path)) == len(fileaslist(norm_text_path)))

        clean_texts = fileaslist(sens_text_path)
        sent_tokens = [sen.split(" ") for sen in fileaslist(norm_text_path)]
        word_counts = torch.LongTensor([[len(tokens) for tokens in sent_tokens]])
        input_length = torch.LongTensor([len(sent_tokens)])
        inputs = namedtuple("Inputs", ["length", "embedding", "word_count"])(
            Variable(input_length), 
            Variable(sentence_embeddings.unsqueeze(0)),
            Variable(word_counts.unsqueeze(2)))
        metadata = namedtuple("Metadata", ["text"])([clean_texts])

        return inputs, metadata 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query", default=os.getenv("QUERY_STR"),
        type=str, required=False)
    parser.add_argument(
        "--model-path", default=os.getenv("RNNSUM_PATH"),
        type=str, required=False)
    parser.add_argument(
        "--folder", type=str, required=True)
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
    args = parser.parse_args()

    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    torch_model = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    
    # initialize the sif embeddings stuff
    (words, We) = em.data_io.getWordmap(args.embd_wordfile_path)
    word2weight = em.data_io.getWordWeight(args.embd_weightfile_path, 1e-3)
    weight4ind = em.data_io.getWeight(words, word2weight)
    embd_params = em.params.params()
    embd_params.rmpc = 0
    sif_model = (words, We, word2weight, weight4ind, embd_params)
    
    # prepare splitta
    splitta_model = sbd.load_sbd_model("../splitta/model_nb/",use_svm=False)

    # create the summarizer
    summarizer = Summarizer(sif_model, torch_model, splitta_model)

    # start the server and listen to summarization requests
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(("", args.port))
    serversocket.listen(5)

    print("Loaded all models successfully, ready to accept requests on %d." % args.port)

    while 1:
      (clientsocket, address) = serversocket.accept()
      data = clientsocket.recv(1000000)
      if str(data, "utf-8") == SUMMARIZATION_TRIGGER:
        # go over all the input files and run summarization
        for input_path in os.listdir(args.folder):
          input_path = args.folder + "/" + input_path
          summary = summarizer.summarize_text(input_path, query=args.query, max_length=args.length)
          output_path = os.path.join(args.summary_dir, os.path.basename(input_path))
          with open(output_path, "w", encoding="utf-8") as fp: fp.write(summary)
          os.chmod(output_path, 0o777)

if __name__ == "__main__":
    main()
