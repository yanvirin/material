import sys, os

duc_dir = sys.argv[1]

def makedps(path):
  inputs_path = path
  parent_path = "/".join(path.split("/")[:-1])
  for d in os.listdir(parent_path):
    targets_path = os.path.join(parent_path, d)
    if "targets" in d and os.path.isdir(targets_path):
      words = int(d.split(".")[1])
      dps_path = targets_path + ".labeled.json"
      os.system("python make_datapoints.py %s %s %s %d" % (inputs_path, targets_path, dps_path, words))

def run(path):
  if path.endswith("inputs"):
    makedps(path)
  else:
    if "targets" not in path:
      if os.path.isdir(path):
        for d in os.listdir(path):
          run(os.path.join(path, d))

run(duc_dir)
