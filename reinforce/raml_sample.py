import json,os,sys,argparse,heapq,time
from itertools import combinations
from multiprocessing import Process, Queue, cpu_count
from eval_rouge import RougeScorer
from collections import defaultdict
from collections import Counter
'''
Holds the needed parameters to run a task
of extracting scored summary candidates
'''
class Task(object):
  def __init__(self, doc_id, sentences, labels):
    self.doc_id = doc_id
    self.sentences = sentences
    self.labels = labels

'''
Implements a beam for objects that you can put
with values and hold only the top k in memory
'''
class Beam(object):
  
  def __init__(self, k):
    self.k = k
    self.data = []
  
  # adds an object with value v onto the beam
  def add(self, o, v):
    if len(self.data) < self.k:
      heapq.heappush(self.data, (v, o))
    else:
      heapq.heappushpop(self.data, (v, o))
  
  # returns all the objects with the values
  def tolist(self):
    if len(self.data) == 0:
      raise Exception("data is empty!")
    res = []
    old_v = -1000
    for i in range(len(self.data)):
      v, o = heapq.heappop(self.data)
      res.append((o, v))
      if v < old_v:
        raise Exception("incosistency v=%f old_v=%f" % (v, old_v))
      old_v = v
    return res

# transforms a list of indices which represents
# a sparse vector into a dense vector of 0's and 1's
def sparse2dense(spl, n):
  res = [0]*n
  for i in spl: res[i] = 1
  return res

# this is the consumer side which takes
# tasks out of the queue and processes them
def work(wid, queue, args, ids2refs, stopwords):
  total = 0
  n = 0
  while True:
    task = queue.get()
    distribution = Counter()
    if task is None:
      break
    st = time.time()
    process(task.doc_id, task.sentences, ids2refs, stopwords, task.labels, args, distribution)
    print("finished process %d, task: %s, time: %f" % (wid, task, time.time() - st))
    n += 1
    total += get_percentile(0.95, distribution)
  queue.put(None)
  print("avg. summaries: %f" % (total/n))

# the method which actually runs the logic of processing
# one document by extracting all possible combinations of sentences
# and computing their rouge scores with the appropriate human refference
def hash_value(value):
  return abs(hash(str(value)))%1000000

def update_count(value, counter):
  value = int(value*1000)
  counter[value] += 1

def get_percentile(percentile, distribution):
  min_v = min(distribution.keys())
  max_v = max(distribution.keys())
  th = (max_v - min_v)*percentile
  s = 0
  for key in distribution:
    if distribution[key] >= th + min_v:
      s += distribution[key]
  return s

def process(doc_id, sentences, ids2refs, stopwords, labels, args, distribution):
  
  top_k = args.top_k
  n = args.n
  length = args.length
  output_dir = args.output_dir

  # keep a beam to put the candidates with the rouge scores on
  beam = Beam(top_k)
  scorer = RougeScorer(length, stopwords)
  ref_paths = ids2refs[doc_id]
  with open(ref_paths[0]) as r:
    ref_dist, word_count = scorer.todist(r.readlines())
  combs = combinations(range(len(sentences)), n)

  q = 0
  s = set()
  for comb in combinations(range(len(sentences)), n):
    summary = [sentences[i] for i in comb]
    assert(len(summary)==n)
    r1 = scorer.eval_rouge(summary, ref_dist, word_count = word_count)
    update_count(r1, distribution)
    if not args.informative or not hash_value(r1) in s:
      q += 1
      beam.add(comb, r1)
      if args.informative: s.add(hash_value(r1))

  # padding with artificial combinations
  if q < top_k:
    for i in range(top_k-q):
      beam.add(tuple(list(range(min(n,len(sentences))))), 0.0)

  # write out the results
  best = beam.tolist()
  assert(len(best)==top_k)
  d = dict()
  d["id"] = doc_id
  d["labels"] = labels
  candidates = []
  for b, v in best:
    candidate = dict()
    candidate["score"] = v
    candidate["labels"] = sparse2dense(b, len(sentences))
    candidates.append(candidate)
  d["label_scores"] = candidates
  os.system("mkdir -p %s" % output_dir)
  with open(output_dir + "/" + doc_id + ".json", "w") as w:
    w.write(json.dumps(d) + "\n")

def collect_ref_paths(references_dir):
  ref_paths = defaultdict(list)
  for f in os.listdir(references_dir):
    ref_id = f.split(".")[0]
    ref_paths[ref_id].append("%s/%s" % (references_dir, f))
  return ref_paths

'''
This script implements producer/consumer pattern for
generating the summary candidates along with the rouge scores
for further analysis and training
'''
if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--inputs-path", type=str, required=True)
  parser.add_argument(
    "--labels-path", type=str, required=True)
  parser.add_argument(
    "--output-dir", type=str, required=True)
  parser.add_argument(
    "--human-abstracts-dir", type=str, required=True)
  # how many sentences in a document to use
  parser.add_argument(
    "--first-k", type=int, required=False, default=25)
  # how many candidates to use
  parser.add_argument(
    "--top-k", type=int, required=False, default=100)
  parser.add_argument(
    "--length", type=int, required=False, default=100)
  # how many sentences  to select for each candidate
  parser.add_argument(
    "--n", type=int, required=False, default=3)
  parser.add_argument(
    "--stopwords", type=str, required=False, default="")
  parser.add_argument(
    "--informative", type=str, required=False, default="False")
  args = parser.parse_args()
  args.informative = args.informative == "True"

  assert(args.n < args.top_k)

  # get the refs dictionary
  ids2refs = collect_ref_paths(args.human_abstracts_dir)
  stopwords = set() if not args.stopwords else set(map(lambda x: x.strip(), open(args.stopwords).readlines()))

  if stopwords:
    print("testing stopwords for 'a' and 'and' inclusion: %s and %s" % ("a" in stopwords, "and" in stopwords))
  
  # crete the pool of workers
  NUMBER_OF_PROCESSES = 5
  queue = Queue(NUMBER_OF_PROCESSES*2)
  workers = [Process(target=work, args=(i, queue, args, ids2refs, stopwords))
                        for i in range(NUMBER_OF_PROCESSES)]
  # start the workers
  for w in workers: w.start()
  
  # serve the work in the producer side of things
  served = 0
  for f in os.listdir(args.inputs_path):
    with open("%s/%s" % (args.inputs_path, f)) as r1, open("%s/%s" % (args.labels_path, f)) as r2:
      d1 = json.load(r1)
      d2 = json.load(r2)
      doc_id = d1["id"]
      assert(d2["id"]==doc_id)
      sentences = []
      assert(len(d1["inputs"])==len(d2["labels"]))
      for input in d1["inputs"][:args.first_k]:
        sentences.append(input["text"])
      task = Task(doc_id, sentences, d2["labels"])
      queue.put(task)
      served += 1
      print("served %d tasks" % served)
  
  queue.put(None)
  for w in workers: w.join()
  print("Finished all jobs, exiting.")
