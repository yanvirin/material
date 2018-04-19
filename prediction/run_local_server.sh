
# params: model_path rescore
source activate env
export MODEL=$1
export RESCORE=$2
export MKL_THREADING_LAYER=GNU
python rnnsum.py --port 9916 --query /tmp/yan/query/queries.txt --folder /tmp/yan/inputs --summary-dir /tmp/yan/outputs --model-path $MODEL --rescore $RESCORE --embd-wordfile-path ../../yan-summarization-docker-2018-03-21/models/glove.840B.300d-freq500K.txt --embd-weightfile-path ../SIF/auxiliary_data/enwiki_vocab_min200.txt
