
# params: model_path rescore
source activate env
MODEL=$1
RESCORE=$2
PORT=$3
export MKL_THREADING_LAYER=GNU
EXTRA=$4
python rnnsum.py --port $PORT --query-folder /tmp/yan/query --folder /tmp/yan/inputs --summary-dir /tmp/yan/outputs --model-path $MODEL --rescore $RESCORE --embd-wordfile-path ../../yan-summarization-docker-2018-06-20/models/glove.840B.300d-freq500K.txt --embd-weightfile-path ../SIF/auxiliary_data/enwiki_vocab_min200.txt $EXTRA
