import sys, os
import argparse

'''
This script runs the make_datapoints.py script on all the duc data input folders
to create labeled json files.
'''

parser = argparse.ArgumentParser(description = 'Create labeled datapoints for all the duc data')
parser.add_argument('duc', metavar="duc", help='the duc data folder')
args = parser.parse_args()

duc_dir = args.duc

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
