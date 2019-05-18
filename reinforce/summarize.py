import sys, re, nnsum, torch, os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pathlib
import ujson as json
from numpy.random import shuffle
MAX_LENGTH = 100

alpha = re.compile("[A-Za-z]")

cleaner = re.compile("\[[a-zA-Z0-9]+\]")
def clean(sen):
    return cleaner.sub("", sen).strip()

splitter = re.compile("[^A-Za-z]+")
def do_tokenize(sen):
    return [token for token in splitter.split(sen) if len(token)>0]

def text2datapoint(text_path):
    with open(text_path) as r: text = r.read()
    d = {"id": "1", "inputs": []}
    for i, sen in enumerate(sent_tokenize(text)):
        sen = clean(sen)
        if len(sen) < 2 or not alpha.search(sen): continue
        input = {}
        sen = sen.lower() if lower_case else sen
        tokens = do_tokenize(sen) if tokenize != "nltk" else word_tokenize(sen)
        input["tokens"] = tokens
        input["text"] = sen
        input["sentence_id"] = i+1
        input["word_count"] = len(tokens) 
        d["inputs"].append(input)

    print("number of sentences: %d" % len(d["inputs"]))
    return d

def load_model(model_path):
    print(" Loading model from {}.".format(model_path))
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.cuda(0)
    print(" Model loaded.")
    return model

def summarize(model, loader):
   with torch.no_grad():
       for step, batch in enumerate(loader, 1):
           batch = batch.to(0)     
           return model.predict(batch, max_length=MAX_LENGTH)

if __name__ == "__main__":
    model1_path = sys.argv[1]
    model2_path = sys.argv[2]
    text_path = sys.argv[3]
    tokenize = sys.argv[4]
    lower_case = sys.argv[5]

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    lower_case = lower_case == "True" or lower_case == "true"

    print(" Summarizing with tokenize with %s and lower case %s\n" % (tokenize, lower_case))
    os.system("mkdir -p /tmp/summarization")
    with open("/tmp/summarization/input.json", "w") as w: json.dump(text2datapoint(text_path), w)
    
    data = nnsum.data.SummarizationDataset(model1.embeddings.vocab, pathlib.Path("/tmp/summarization/"))
    loader = nnsum.data.SummarizationDataLoader(data, batch_size=1, num_workers=1)

    summary1 = summarize(model1, nnsum.data.SummarizationDataLoader(data, batch_size=1, num_workers=1))
    summary2 = summarize(model2, nnsum.data.SummarizationDataLoader(data, batch_size=1, num_workers=1))
   
    print("==================================")
    print("\n".join(summary1[0]))
    print("==================================")
    print("\n".join(summary2[0]))

