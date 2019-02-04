import os, json, sys, time
from multiprocessing import Process, Queue, Lock
import rouge_papier

TEMP_BASE = "adsdscsdSDDDDSDCSDcsdcaghcsgckdjnkerh424r2"

def write_content(content, path):
  with open(path, "w") as w:
    w.write(content)

def temp_filename(suffix):
  return "/dev/shm/"+TEMP_BASE+"_"+str(os.getpid())+"_"+suffix

def clean_tempfiles():
  os.system("rm /dev/shm/"+TEMP_BASE+"*")

def create_labels(queue):

  config_path = temp_filename(".config")
  summary_path = temp_filename(".summary")

  while True:
    try:
       doc_id, summary_length, inputs_dir, human_abstracts_dir, labels_dir = queue.get()
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
             summary = "\n".join([x[1] for x in new_best])
             write_content(summary, summary_path)
             path_data.append([summary_path, ["%s/%s.spl" % (human_abstracts_dir, doc_id)]])
             config_text = rouge_papier.util.make_simple_config_text(path_data)
             write_content(config_text, config_path)
             df = rouge_papier.compute_rouge(
                 config_path, max_ngram=2, lcs=True,
                 remove_stopwords=True, length=summary_length)
             r1, _, _ = df.values[0].tolist()
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

  JOBS = 10
 
  inputs_dir = sys.argv[1]
  human_abstracts_dir = sys.argv[2]
  labels_dir = sys.argv[3]
  summary_length = int(sys.argv[4])

  os.system("mkdir -p %s" % labels_dir)

  queue = Queue(100)

  for i in range(JOBS):
    p = Process(target=create_labels, args=[queue])
    p.daemon = True
    p.start()

  files = os.listdir(inputs_dir)
  for f in files:
    doc_id, ext = f.split(".")
    queue.put((doc_id, summary_length, inputs_dir, human_abstracts_dir, labels_dir))

  while(not queue.empty()):
    print("waiting for consumers to finish...")
    time.sleep(5)

  clean_tempfiles()
