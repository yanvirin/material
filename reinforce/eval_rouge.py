from collections import defaultdict

class RougeScorer(object):
  
  def __init__(self, max_words, stopwords = set()):
    self.max_words = max_words
    self.stopwords = stopwords

  def todist(self, sens):
    words = 0
    d = defaultdict(int)
    for sen in sens:
      for word in sen.lower().split(" "):
        words += 1   
        if word in self.stopwords: continue
        d[word] += 1
        if words >= self.max_words: break
      if words >= self.max_words: break
    return d, words

  def eval_rouge(self, sys_sens, ref_sens, word_count = 0):
    if type(ref_sens) is dict:
      ref_dist = ref_sens
      ref_words = word_count
      assert(ref_words > 0)
    else:
      ref_dist, ref_words = self.todist(ref_sens)
    sys_dist, _ = self.todist(sys_sens)
    covered = 0
    for key in ref_dist:
      covered += min(ref_dist[key], sys_dist[key])
    return float(covered) / ref_words
