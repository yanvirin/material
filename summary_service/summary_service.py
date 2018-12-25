import logging
import pathlib
import argparse
import socket
import os
import traceback
import message_handlers as mh
import json
import run_compressor

def read_summarizer_conf(working_dir):
  conf = None
  for f in os.listdir(working_dir):
    if f.endswith(".json"):
      with open("%s/%s" % (working_dir, f)) as r:
        conf = json.load(r)["summarizer"]
        break
  if not conf: 
    raise Exception("no configuration json file found in %s" % working_dir)
  return conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", required=True, type=int)
    parser.add_argument(
        "--query-processor", type=str, required=True)
    parser.add_argument(
        "--clir-results", type=str, required=True)
    parser.add_argument(
        "--nist-data", type=str, required=True)
    parser.add_argument(
        "--length", default=100, type=int)
    parser.add_argument(
        "--summary-dir", required=True, type=str)
    parser.add_argument(
        "--english-embeddings", required=True, type=str)
    parser.add_argument(
        "--english-counts", required=True, type=str)
    parser.add_argument(
        "--english-stopwords", required=True, type=str)
    parser.add_argument(
        "--source-embeddings-dir", required=True, type=str)
    parser.add_argument(
        "--working-dir", required=True, type=str)
    parser.add_argument(
        "--topic-model-path", required=False, type=str, default=None)
    parser.add_argument("--use-topics", action="store_true", default=False)
    parser.add_argument(
        "--max-topic-words", required=False, type=int, default=5)
    parser.add_argument(
        "--topic-cache", type=str, default=None, required=False)
    parser.add_argument(
        "--use-topic-cache", action="store_true", default=False)
    parser.add_argument(
        "--use-compressor", action="store_true", default=False)
    parser.add_argument(
        "--separated", action="store_true", default=False)
    parser.add_argument(
        "--split", action="store_true", default=False)
    parser.add_argument(
        "--compressor-embedding-lookup", type=str, required=False)
    parser.add_argument(
        "--compressor-model", type=str, required=False)
    parser.add_argument(
        "--logging-level", required=False, type=str, default="warning",
        choices=["info", "warning", "debug"])
    parser.add_argument("--sentence-rankers", default=["translation"], 
        choices=["translation", "source", "crosslingual", 
                 "lexical-expansion-translation", "qa"], type=str,
        nargs="+")
    parser.add_argument("--qa-question-words", default=["what"],
        choices=["what", "where", "when", "what-kind", "why", "how", "which"], type=str,
        nargs="+")

    args = parser.parse_args()

    # find configuration in the working-dir and read the summarizer object
    conf = read_summarizer_conf(args.working_dir)

    source_embeddings = "%s/%s.vec" % (args.source_embeddings_dir, conf["language"])
    source_counts = "%s/%s.freq" % (args.source_embeddings_dir, conf["language"])

    logging.getLogger().setLevel(logging.__dict__[args.logging_level.upper()])

    summary_dir = pathlib.Path(args.summary_dir)
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True, exist_ok=True)

    if args.topic_model_path:
        topic_model_path = pathlib.Path(args.topic_model_path) 
    else:
        topic_model_path = None
    system_context = {
        "separated": args.separated,
        "split": args.split,
        "query_processor_path": pathlib.Path(args.query_processor),
        "clir_results_path": pathlib.Path(args.clir_results),
        "nist_data": pathlib.Path(args.nist_data),
        "translation": {
            "text": conf["mt_text_version"],
            "audio": conf["mt_audio_version"].replace("[lang]",conf["language"])
        },
        "morphology": {
            "text":  conf["morph_text_version"],
            "audio": conf["morph_audio_version"].replace("[lang]",conf["language"]),
        },
        "topic_model": {
            "path": topic_model_path,
            "use_topic_model": args.use_topics,
            "model": None,
            "max_topic_words": args.max_topic_words,
            "topic_cache_path": pathlib.Path(args.topic_cache) if args.topic_cache else None,
            "topic_cache": None,
            "use_topic_cache": args.use_topic_cache,
        },
        "compressor": {
            "use_compressor": args.use_compressor,
            "compressor_embedding_lookup": args.compressor_embedding_lookup,
            "compressor_model_path": args.compressor_model
        },
        "english_embeddings": {
            "path": pathlib.Path(args.english_embeddings),
            "counts": pathlib.Path(args.english_counts),
            "model": None,
        },
        "source_embeddings": {
            "path": pathlib.Path(source_embeddings),
            "counts": pathlib.Path(source_counts),
            "model": None,
        },
        "sentence_rankers": {
            "translation": "translation" in args.sentence_rankers,
            "source": "source" in args.sentence_rankers,
            "crosslingual": "crosslingual" in args.sentence_rankers,
            "lexical-expansion-translation": "lexical-expansion-translation" in args.sentence_rankers,
            "qa": "qa" in args.sentence_rankers,
        },
        "qa_question_words": args.qa_question_words,
        "summary_length": args.length,
        "summary_dir": summary_dir,
        "english_stopwords": {
            "path": pathlib.Path(args.english_stopwords),
            "model": None,
        },
    }
    
    if system_context["topic_model"]["use_topic_cache"]:
        if system_context["topic_model"]["topic_cache_path"].exists():
            system_context["topic_model"]["topic_cache"] = json.loads(
                system_context["topic_model"]["topic_cache_path"].read_bytes(),
                encoding="utf8")
        else:
            system_context["topic_model"]["topic_cache"] = {}

    mh.handle_source_embeddings(
        system_context["source_embeddings"], system_context)
    mh.handle_english_stopwords(
        system_context["english_stopwords"], system_context)
    mh.handle_english_embeddings(
        system_context["english_embeddings"], system_context)
    mh.handle_lda(
        system_context["topic_model"], system_context)
    mh.handle_compressor(system_context["compressor"], system_context)

    print("Topic Model:")
    for k, v in system_context["topic_model"].items():
        if k != "topic_cache":
            print(k, v)
    print("Sentence Rankers:")
    print(system_context["sentence_rankers"])
    print("Loaded server on port {} ...".format(args.port))

    # start the server and listen to summarization requests
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind(("", args.port))
    serversocket.listen(5)

    while 1:
        (clientsocket, address) = serversocket.accept()
        data = clientsocket.recv(1000000)
        try:
            params = json.loads(str(data, "utf-8"))
    
            if params["message_type"] == "query":
                mh.handle_query(params["message_data"], system_context)
            elif params["message_type"] == "reload_module":
                mh.handle_reload_module(params["message_data"], system_context)
            elif params["message_type"] == "lda":
                mh.handle_lda(params["message_data"], system_context)
            elif params["message_type"] == "english_embeddings":
                mh.handle_english_embeddings(
                    params["message_data"], system_context)
            elif params["message_type"] == "source_embeddings":
                mh.handle_source_embeddings(
                    params["message_data"], system_context)
            elif params["message_type"] == "example":
                mh.summary_handler.summarize_example(
                    params["message_data"], system_context)
            elif params["message_type"] == "topic_cache":
                mh.handle_topic_cache(params["message_data"], system_context)
            elif params["message_type"] == "logging":
                mh.handle_logging(params["message_data"], system_context)

            clientsocket.send(b"OK")
        except Exception as e:
            print("\n")
            traceback.print_exc()
            clientsocket.send(b"ERROR")

if __name__ == "__main__":
    main()
