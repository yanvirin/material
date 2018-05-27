import sys,os,math
from torch.nn.modules.distance import CosineSimilarity as cosim
import numpy as np
sys.path.append("../splitta")
from word_tokenize import normalize

class SimilarityExtractor:

  def __init__(self, use_text_cosine, use_embd_cosine):
    self.use_text_cosine = use_text_cosine
    self.use_embd_cosine = use_embd_cosine

  def get_embd_scores(self, inputs):
    embedding = inputs.embedding.data
    # narrow the input tensor
    sen_embeds = embedding.narrow(2, 0, 300)
    query_embeds = embedding.narrow(2, 300, 300)

    # compute the cosine similarity
    cos = cosim(dim=2)
    cos_scores = cos(sen_embeds, query_embeds)
    return cos_scores

  def inner(self, toks1, toks2):
    score = 0
    for t in toks1:
      if t in toks2: score += 1
    return float(score)
  
  def text_cosine(self, toks1, toks2):
    raw = self.inner(toks1,toks2) / (math.sqrt(self.inner(toks1,toks1))*math.sqrt(self.inner(toks2,toks2)))
    return 2*raw - 1
    
  def get_text_scores(self, metadata):
    batches = []
    for b in range(len(metadata.text)):
      scores = []
      query_toks = normalize(metadata.query[b]).split(" ")
      for i in range(len(metadata.text[b])):
        text_toks = normalize(metadata.text[b][i]).split(" ")
        scores.append(self.text_cosine(query_toks,text_toks))   
      batches.append(scores)
    return batches 

  def extract(self, inputs, metadata, word_limit, rescore):
    summaries = []
    embd_scores = self.get_embd_scores(inputs)
    text_scores = self.get_text_scores(metadata)
    final_scores = []
    for b in range(len(metadata.text)):
      scores = []
      for i in range(len(metadata.text[b])):
        score = 0.0
        dom = 0
        if self.use_text_cosine:
          score += text_scores[b][i]
          dom += 1
        if self.use_embd_cosine:
          score += embd_scores[b][i]
          dom += 1
        score = score / dom
        scores.append(score)
      final_scores.append(scores)
    
    for b in range(len(metadata.text)):
      summary = []
      indices = np.argsort([-s for s in final_scores[b]])
      count = 0
      for i in indices:
        candidate = metadata.text[b][i]
        c = len(candidate.split(" "))
        if count + c <= word_limit:
          summary.append(candidate)
          count += c
        if count + c == word_limit: break
      summaries.append("\n".join(summary))
      
    return summaries, None

if __name__ == "__main__":
  extractor = SimilarityExtractor(False,False)
  score = extractor.text_cosine(sys.argv[1].split(" "), sys.argv[2].split(" "))
  print(score)
