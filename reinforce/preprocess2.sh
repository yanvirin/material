
data_dir=$1

set -e

python concat_in_order_sds.py $data_dir/clean-docsets $data_dir/sds-full-in-order
python normalize_tokenize.py $data_dir/sds-full-in-order $data_dir/sds-full-in-order-norm
python normalize_tokenize.py $data_dir/human-abstracts $data_dir/human-abstracts-norm
python convert_duc_sds.py $data_dir/sds-full-in-order-norm $data_dir/sds-full-in-order-inputs
python extract_ml_labels.py $data_dir/sds-full-in-order-inputs $data_dir/human-abstracts-norm $data_dir/sds-full-in-order-labels 100
python raml_sample.py --inputs-path $data_dir/sds-full-in-order-inputs --labels-path $data_dir/sds-full-in-order-labels --human-abstracts-dir $data_dir/human-abstracts-norm --output-dir $data_dir/sds-full-in-order-raml-labels --first-k 1000 --informative False
