import argparse
import pathlib
import json
import results_handler
import sys
import lda
from multiprocessing import Pool
import summary_handler as sh


def init_worker(model_path):
    global model
    print("Loading model")
    model = lda.load_topic_model(model_path)
    print("Loaded!")

def worker(args):
    global model
    result, query_data, system_context = args
    query_id = query_data["parsed_query"][0]["info"]["queryid"]
    doc_id = result["doc_id"]

    doc_translation = sh.read_translation(result["translation_path"])
    query_content = sh.get_query_content(query_data["IARPA_query"])

    return "{}-{}".format(query_id, result["doc_id"]), lda.get_topics(
        model, doc_translation, query_content,
        5,
        always_highlight=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nist-data", type=str, required=True)
    parser.add_argument(
        "--query-processor", type=str, required=True, nargs="+")
    parser.add_argument(
        "--clir-results", type=str, required=True, nargs="+")
    parser.add_argument(
        "--topic-model-path", required=False, type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()
    assert len(args.query_processor) == len(args.clir_results)

    arguments = []

    for qp_dir, ec_dir in zip(args.query_processor, args.clir_results):
        
        for qp_path in pathlib.Path(qp_dir).glob("*"):

            system_context = {
                "query_processor_path": pathlib.Path(qp_dir),
                "clir_results_path": pathlib.Path(ec_dir),
                "nist_data": pathlib.Path(args.nist_data),
                "translation": {
                    "text": {
                        "1A": "umd-nmt-v2.1_sent-split-v2.0",
                        "1B": "umd-nmt-v2.1_sent-split-v2.0"
                    },
                    "audio": {
                        "1A": "umd-nmt-v2.1_material-asr-sw-v5.0/",
                        "1B": "umd-nmt-v2.1_material-asr-tl-v5.0/"
                    },
                },
                "morphology": {
                    "text": {
                        "1A": "material-scripts-morph-v3.0_cu-code-switching-v3.0_sent-split-v2.0",
                        "1B": "material-scripts-morph-v3.0_cu-code-switching-v3.0_sent-split-v2.0",
                    },
                    "audio": {
                        "1A": "material-scripts-morph-v3.0_material-asr-sw-v5.0/",
                        "1B": "material-scripts-morph-v3.0_material-asr-tl-v5.0/",
                    },
                },
        
            }
        
            query_data = json.loads(qp_path.read_bytes(), encoding="utf8")
            query_id = query_data["query_id"]
            clir_results_tsv = pathlib.Path(ec_dir) / "{}.tsv".format(
                query_id)

            results = results_handler.load_clir_results(
                clir_results_tsv, 
                system_context)

            for result in results:
                arguments.append((result, query_data, system_context))

    pool = Pool(6, initializer=init_worker, initargs=[args.topic_model_path])


    cache = {}
    for i, result in enumerate(pool.imap_unordered(worker, arguments), 1):
        sys.stdout.write("{}/{}\r".format(i, len(arguments)))
        sys.stdout.flush()
        cache[result[0]] = result[1]

    print()
    
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf8") as fp:
        fp.write(json.dumps(cache))

if __name__ == "__main__":
    main()
