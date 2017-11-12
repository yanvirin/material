import sys, os
from os.path import join
import numpy as np
import random

random.seed(1000)

dir = sys.argv[1]
dist = sys.argv[2] == "dist"

def read_vectors(sen_file, emb_file):
  sentences = []
  vectors = []

  for line in open(sen_file, "r"):
    sentences.append(line.strip())

  for line in open(emb_file, "r"):
    numbers = line.strip().split(" ")
    floats = map(float, numbers)
    vectors.append(np.array(floats))

  return sentences, vectors

def distance(v1, v2, dist=True):
  if dist:
    return np.linalg.norm(v1-v2)
  else:
    return 1-v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

n = 0
for d in os.listdir(dir):
  
  if n == 100: break
  n += 1
  
  curr_path = os.path.join(dir, d)
  # read the content
  (cont_sens, cont_vecs) = read_vectors(join(curr_path, "content.txt.nrm"), join(curr_path, "content.txt.emd"))

  assert(len(cont_sens) == len(cont_vecs))

  # choose a sentence
  candidate_idx = []
  for i in range(0, len(cont_sens)):
    if len(cont_sens[i].split(" ")) > 5: candidate_idx.append(i)
  idx = random.sample(candidate_idx, 1)[0]
  sentence = cont_sens[idx]
  vector = cont_vecs[idx]

  # read all the sources
  sources = join(curr_path, "sources")
  (source_sens, source_vecs) = ([], [])
  for s in os.listdir(sources):
    if s.endswith(".txt"):
      (source_sen_temp, source_vec_temp) = read_vectors(join(sources, s)+".nrm", join(sources, s)+".emd")
      source_sens.extend(source_sen_temp)
      source_vecs.extend(source_vec_temp)
  
  assert(len(source_sens) == len(source_vecs))

  # sort the source vectors
  scores = []
  for i in range(0, len(source_vecs)):
    scores.append((i, distance(vector, source_vecs[i], dist=dist)))
  sorted_scores = sorted(scores, key=lambda x:x[1])
  
  print "*******************************************************"
  print
  print sentence
  print
  print "*******************************************************"
  print "============================"
  for i,s in sorted_scores[:10]:
    print source_sens[i]
    print "=========================="
