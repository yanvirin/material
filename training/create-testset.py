import os,sys,random

sys.path.append("../rouge-scripts")
import rouge as rge

dataset_path = sys.argv[1]
testset_path = sys.argv[2]

os.system("mkdir -p %s" % testset_path)

score_thresh = float(sys.argv[3])
test_ratio = float(sys.argv[4])

testset = []
for d in os.listdir(dataset_path):
  if d == ".README": continue
  if random.random() < test_ratio:
    best_score = 0.0
    for f in os.listdir(os.path.join(dataset_path, d, "sources")):
      if f.endswith(".rge"):
        score = rge.parse_rouge(os.path.join(dataset_path, d, "sources", f), 2)
        if score > best_score:
          best_score = score
    if best_score > score_thresh: testset.append(d)
    print "article with %f found." % best_score

print "creating %d test articles" % len(testset)

for d in testset:
  os.system("mv %s %s" % (os.path.join(dataset_path, d), os.path.join(testset_path, d)))



    
