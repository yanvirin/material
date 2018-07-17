
# params: model_path rescore
source activate env
LANGUAGE=$1
PORT=$2
QUERY=$3
RESULTS=$4
HIGHLIGHT=False
TRANSLATE=True
export MKL_THREADING_LAYER=GNU
python rnnsum.py --port $PORT --segment False --results $RESULTS --query-folder $QUERY --folder /storage/data/NIST-data --summary-dir /tmp/yan/outputs --model-path hru --rescore False --embd-wordfile-path ../../yan-summarization-docker-2018-06-20/models/en_${LANGUAGE}_embds.txt --embd-weightfile-path ../../yan-summarization-docker-2018-06-20/models/en_${LANGUAGE}_freq.txt --text-similarity=False --embd-similarity=True --gen-image=True --translate-query=$TRANSLATE --highlight $HIGHLIGHT --language=$LANGUAGE
