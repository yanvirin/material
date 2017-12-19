# A summarization project, Columbia University

## How to compute baselines?

In the **training** folder you will find the baseline.py script.

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

## How to convert a dataset to a labeled data set for training/evaluation?

In the **training** folder you will find the convert_dataset.py script.

usage: convert_dataset.py [-h] ds out ver mrge typ

Convert a dataset to a labeled dataset for training or evaluation

positional arguments:
  ds          the path to the dataset top level folder
  out         the output file with the labedled data points as a json file
  ver         whether to use .best1 or .best2 to create the datapoints
  mrge        the minimal ROUGE2 in the .rge file to consider as valid
  typ         to create mds or sds datapoints (mds, sds)

example:

python convert_dataset.py ../../material-data/dataset /tmp/traininig.json 1 0.1 msd

## How to create labeled datapoints in json format from DUC data?

In the **training** folder there is a run_on_duc.py script.

usage: run_on_duc.py [-h] duc

Create labeled datapoints for all the duc data

positional arguments:
  duc         the duc data folder

example:

python run_on_duc.py ../../material-data/duc_data

## How to calculate the distribution of rouge scores accross the dataset?

In the **rouge_scripts** folder there is a calc_dist.py script.

usage: calc_dist.py [-h] ds

Calculate the distributions of ROUGE 1 and 2 scores

positional arguments:
  ds          the path to one input duc data folder

example:

python calc_dist.py ../../material-data/dataset

## How to extract the best sentences according to IGR or SGR?

In **rouge_scripts** folder there is the extract_best_sentences.py which should be
run on all articles from the dataset with "parallel" command:

ls dataset | parallel -j 10 'python extract_best_sentences.py dataset 1 {} 1'


