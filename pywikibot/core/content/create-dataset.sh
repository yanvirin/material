sources_dir=$1
dataset_dir=$2

python create-dataset.py $sources_dir $dataset_dir

python clean-dataset.py $dataset_dir
