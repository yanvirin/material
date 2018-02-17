import json
import math

'''
This file holds all kind of utils that are used throught the code base.
'''

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    score = sumxy/math.sqrt(sumxx*sumyy)
    return score

'''
Returns an average for the list of vectors (represented as lists of floats)
'''
def average(vs):
  v = vs[0]
  for i in range(1, len(vs)):
    for j in range(len(v)):
      v[j] += vs[i][j]
  for j in range(len(v)): v[j] = v[j] / len(vs)
  return v

def fileaslist(f):
  with open(f, 'r') as fh: return [line.strip() for line in fh.readlines()]

def write2file(text, f):
  with open(f, "w") as fw: fw.write(text)

'''
Sorts the json datapoint results
'''
def sort_results(results):
  for r in results:
    r.sort(cmp=lambda x,y: cmp(x["docset_id"],y["docset_id"])*100+cmp(x["doc_id"],y["doc_id"])*10+cmp(int(x["sentence_id"]),int(y["sentence_id"])))
  results.sort(cmp=lambda x,y: cmp(x[0]["docset_id"],y[0]["docset_id"]))

'''
Saves the json datapoint results
'''
def save_results(results, output_file):
  sort_results(results)

  # write the results out
  with open(output_file, 'w') as outfile:
    for i,r in enumerate(results):
      json.dump(r, outfile)
      if i != len(results)-1: outfile.write("\n")
