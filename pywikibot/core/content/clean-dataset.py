import sys, os

dataset_dir = sys.argv[1]

for d in os.listdir(dataset_dir):
  if not os.path.exists("%s/%s/%s" % (dataset_dir, d, "content.txt")) or len(os.listdir("%s/%s/%s" % (dataset_dir, d, "sources"))) == 0:
    os.system("rm -rf %s/%s" % (dataset_dir, d))
