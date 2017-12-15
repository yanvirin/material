# A summarization project, Columbia University

## How to compute baselines?

In the folder training you will find the baseline.py script.

usage: baseline.py [-h] ds ver out mrge typ

positional arguments:
  ds          the path to the dataset top level folder
  ver         ROUGE ver to user either 1 or 2 for choosing the best source to
              use (mds)
  out         the path of the output ROUGE file; should be global path
  mrge        the minimal ROUGE2 in the .rge file to consider as valid
  typ         the type of baseline to run: centroid, first3, rand3

example:

python baseline.py ../../material-data/dataset 1 ../training/baseline.rand3 0.1 rand3


