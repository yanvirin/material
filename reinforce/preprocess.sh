
data_dir=$1

set -e

python concat_shuffle_sds.py $data_dir/clean-docsets $data_dir/sds-full-shuffled
python normalize_tokenize.py $data_dir/sds-full-shuffled $data_dir/sds-full-shuffled-norm
python normalize_tokenize.py $data_dir/human-abstracts $data_dir/human-abstracts-norm
python convert_duc_sds.py $data_dir/sds-full-shuffled-norm $data_dir/sds-full-shuffled-inputs
python extract_ml_labels.py $data_dir/sds-full-shuffled-inputs $data_dir/human-abstracts-norm $data_dir/sds-full-shuffled-labels 100
python raml_sample.py --inputs-path $data_dir/sds-full-shuffled-inputs --labels-path $data_dir/sds-full-shuffled-labels --human-abstracts-dir $data_dir/human-abstracts-norm --output-dir $data_dir/sds-full-shuffled-raml-labels --first-k 1000 --informative True
