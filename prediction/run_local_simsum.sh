
# params: model_path rescore
source activate env
CONFIGURATION=$1
PORT=$2
QUERY=$3
RESULTS=$4
HIGHLIGHT=False
export MKL_THREADING_LAYER=GNU
python3 simsum.py --port $PORT --results $RESULTS --query-folder $QUERY --folder /storage2/data/NIST-data --summary-dir ../../tmp/outputs --embds-dir ../../yan-summarization-docker-2018-07-20/models --text-similarity=False --embd-similarity=True --gen-image=True --highlight $HIGHLIGHT --topic-model-path ../../yan-summarization-docker-2018-07-20/models/nyt-lda-model --debug True
