import logging
import summary_handler
import word_embeddings
import lda
import importlib
import results_handler
import json
import pathlib
import os


def handle_logging(request_data, system_context):
    logging.getLogger().setLevel(logging.__dict__[request_data["level"]])

def handle_english_stopwords(request_data, system_context):
    if request_data["path"]:
        path = pathlib.Path(request_data["path"])
        if path != system_context["english_stopwords"]["path"] or \
                system_context["english_stopwords"]["model"] is None:
            print("Loading English Stopwords: {}".format(
                path))
            with open(path, "r", encoding="utf8") as fp:
                model = set([l.strip() for l in fp]) 
            system_context["english_stopwords"]["model"] = model
            print("Loading English Stopwords: done!") 

def handle_english_embeddings(request_data, system_context):
    if request_data["path"]:
        path = pathlib.Path(request_data["path"])
        counts = pathlib.Path(request_data["counts"])
        if path != system_context["english_embeddings"]["path"] or \
                counts != system_context["english_embeddings"]["counts"] or \
                system_context["english_embeddings"]["model"] is None:
            print("Loading English Embedings:\n  emb={}\n  counts={}".format(
                path, counts)) 
            model = word_embeddings.load_embeddings(path, counts_path=counts)
            system_context["english_embeddings"]["model"] = model
            print("Loading English Embedings: done!") 

def handle_tagalog_embeddings(request_data, system_context):
    if request_data["path"]:
        path = pathlib.Path(request_data["path"])
        counts = pathlib.Path(request_data["counts"])
        if path != system_context["tagalog_embeddings"]["path"] or \
                counts != system_context["tagalog_embeddings"]["counts"] or \
                system_context["tagalog_embeddings"]["model"] is None:
            print("Loading Tagalog Embedings:\n  emb={}\n  counts={}".format(
                path, counts)) 
            model = word_embeddings.load_embeddings(path, counts_path=counts)
            system_context["tagalog_embeddings"]["model"] = model
            print("Loading Tagalog Embedings: done!") 

def handle_swahili_embeddings(request_data, system_context):
    if request_data["path"]:
        path = pathlib.Path(request_data["path"])
        counts = pathlib.Path(request_data["counts"])
        if path != system_context["swahili_embeddings"]["path"] or \
                counts != system_context["swahili_embeddings"]["counts"] or \
                system_context["swahili_embeddings"]["model"] is None:
            print("Loading Swahili Embedings:\n  emb={}\n  counts={}".format(
                path, counts)) 
            model = word_embeddings.load_embeddings(path, counts_path=counts)
            system_context["swahili_embeddings"]["model"] = model
            print("Loading Swahili Embedings: done!") 

def handle_lda(request_data, system_context):
    if request_data["path"]:
        path = pathlib.Path(request_data["path"])
        if path != system_context["topic_model"]["path"] or \
                    system_context["topic_model"]["model"] is None:
            system_context["topic_model"]["path"] = path
            print("Loading topic model: {}".format(path))
            system_context["topic_model"]["model"] = lda.load_topic_model(
                path)
            print(system_context["topic_model"]["model"])
            print("Loading topic model: done!")
    if request_data["use_topic_model"] is not None:
        system_context["topic_model"]["use_topic_model"] = request_data[
            "use_topic_model"]
        print("Using Topic Model:", 
            system_context["topic_model"]["use_topic_model"])
    if request_data["max_topic_words"] is not None:
        system_context["topic_model"]["max_topic_words"] = request_data[
            "max_topic_words"]
        print("Using Max Topic Words:", request_data["max_topic_words"])

def handle_topic_cache(request_data, system_context):
    if request_data["action"] == "save" and \
            system_context["topic_model"]["topic_cache_path"]:
        path = system_context["topic_model"]["topic_cache_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Saving lda topic cache to: {}".format(path))
        path.write_text(
            json.dumps(system_context["topic_model"]["topic_cache"]), 
            encoding="utf8")

def handle_reload_module(request_data, system_context):
    module = request_data["module"]
    print("Reloading module:", module)
    eval("importlib.reload(importlib.import_module(\"{}\"))".format(module))
    print("done!")

def handle_query(request_data, system_context):
    query_id = request_data["query_id"]
    logging.info(" summarizing results for {}".format(query_id))

    query_processor_path = system_context["query_processor_path"] / query_id
    with open(query_processor_path, "r", encoding="utf8") as fp:
        query_data = json.load(fp)

    clir_results_tsv = system_context["clir_results_path"] / "q-{}.tsv".format(
        query_id)

    results = results_handler.load_clir_results(
        clir_results_tsv, system_context)

    query_dir = system_context["summary_dir"] / query_id
    query_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        summary_handler.summarize_query_result(
            result, query_data, system_context)

    os.system("chmod -R 777 {}".format(str(query_dir)))

