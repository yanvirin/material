import os, json, sys, time
from multiprocessing import Process, Queue, Lock
from eval_rouge import RougeScorer

def create_labels(queue):
  while True:
    try:
       doc_id, summary_length, inputs_dir, human_abstracts_dir, labels_dir, scorer = queue.get()
       with open("%s/%s.json" % (inputs_dir, doc_id)) as r: d = json.load(r)
       inputs = d["inputs"]
       best = []
       length = 0
       best_score = 0
       while True:
         best_candidate = None
         best_candidate_length = 0
         for i, input in enumerate(inputs):
           candidate = (i, input["text"])
           candidate_length = input["word_count"]
           if candidate not in best and length + candidate_length <= summary_length:
             path_data = []
             new_best = best + [candidate]
             summary = [x[1] for x in new_best]
             abstracts_paths = ["%s/%s"%(human_abstracts_dir,f) for f in os.listdir(human_abstracts_dir) if f.startswith(doc_id+".")]
             r1 = 0.0
             for p in abstracts_paths:
               with open(p) as r:
                 abstract = [l.strip() for l in r.readlines()]
                 r1 += scorer.eval_rouge(summary, abstract)/len(abstracts_paths)
             if r1 > best_score:
               best_score = r1
               best_candidate = candidate
               best_candidate_length = candidate_length
         if not best_candidate: break
         best.append(best_candidate)
         length += best_candidate_length
          
       print("Computed best for doc_id: %s with %d candidates and final score: %f, with total length: %d" % (
              doc_id, len(best), best_score, length))
       best_idx = set([x[0] for x in best])
       labels = [1 if i in best_idx else 0 for i,x in enumerate(inputs)]
       ld = {"labels": labels, "id": doc_id}
       with open("%s/%s.json" % (labels_dir,doc_id), "w") as w:
         json.dump(ld, w)
    except Exception as e:
      print(e)

if __name__ == "__main__":

  JOBS = 4
 
  inputs_dir = sys.argv[1]
  human_abstracts_dir = sys.argv[2]
  labels_dir = sys.argv[3]
  summary_length = int(sys.argv[4])

  scorer = RougeScorer(max_words = summary_length)
  
  os.system("mkdir -p %s" % labels_dir)

  queue = Queue(5)

  for i in range(JOBS):
    p = Process(target=create_labels, args=[queue])
    p.daemon = True
    p.start()

  files = os.listdir(inputs_dir)
  for f in files:
    doc_id, ext = f.split(".")
    queue.put((doc_id, summary_length, inputs_dir, human_abstracts_dir, labels_dir, scorer))

  while(not queue.empty()):
    print("waiting for consumers to finish...")
    time.sleep(5)
