import argparse
import json
import os
import sys
import rouge_papier
import random


def make_lead(example, limit=100):
    size = 0
    summary_texts = []

    for input in example["inputs"]:
        summary_texts.append(input["text"])
        size += len(input["text"].split())
        if size >= limit:
            break
    return "\n".join(summary_texts)

def make_random(example, limit=100):
    random.seed(457345)
    size = 0
    summary_texts = []

    input_texts = [input["text"] for input in example["inputs"]]
    random.shuffle(input_texts)
    for input in input_texts:
        summary_texts.append(input)
        size += len(input.split())
        if size >= limit:
            break
    return "\n".join(summary_texts)

def make_tail(example, limit=100):
    size = 0
    summary_texts = []

    for input in example["inputs"][::-1]:
        summary_texts = [input["text"]] + summary_texts 
        size += len(input["text"].split())
        if size >= limit:
            break
    return "\n".join(summary_texts)
 
 
def find_references(ref_summary_dir, doc_id):
    paths = []
    for fn in os.listdir(ref_summary_dir):
        ref_id = fn.split(".")[0]
        if ref_id == doc_id:
            paths.append(os.path.join(ref_summary_dir, fn))
    return paths

def evaluate_method(inputs_path, abs_dir, output_dir, 
                    method="lead", summary_length=100):

    ids = []
    rouge_config_paths = []
    for f in os.listdir(inputs_path):
        with open("%s/%s"%(inputs_path,f), "r") as inp_fp:
            example = json.load(inp_fp)
            if method == "lead":
                sys_summary_text = make_lead(example, limit=summary_length)
            elif method == "tail":
                sys_summary_text = make_tail(example, limit=summary_length)
            elif method == "random":
                sys_summary_text = make_random(example, limit=summary_length)
            else:
                raise Exception("method not implemented: " + method)
            
            ref_paths = find_references(abs_dir, example["id"])
            sys_path = os.path.join(
                output_dir, "{}.summary".format(example["id"]))
            with open(sys_path, "w") as out_fp:
                out_fp.write(sys_summary_text)
            rouge_config_paths.append([sys_path, ref_paths])
            ids.append(example["id"])
    with rouge_papier.util.TempFileManager() as manager:
        config_text = rouge_papier.util.make_simple_config_text(
                rouge_config_paths)
        config_path = manager.create_temp_file(config_text)
        df, conf = rouge_papier.compute_rouge(
                config_path, max_ngram=2, lcs=True, 
                remove_stopwords=False,
                length=summary_length, return_conf=True)
        df.index = ids + ["average"]
            #df = pd.concat([df[:-1].sort_index(), df[-1:]], axis=0)
        return df, conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-abs", type=str, required=True)
    parser.add_argument("--test-inputs", type=str, required=True)
    parser.add_argument("--test-abs", type=str, required=True)
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--summary-length", type=int, default=100)
    parser.add_argument("--method", type=str,
                        choices=["lead", "tail", "random"], default="lead")

    args = parser.parse_args()
    os.system("mkdir -p %s/%s" % (args.eval_dir, "eval-results"))

    valid_summary_dir = os.path.join(
        args.eval_dir, "summaries", args.method, "valid")

    if not os.path.exists(valid_summary_dir):
        os.makedirs(valid_summary_dir)

    valid_rouge_df, valid_rouge_conf = evaluate_method(
        args.valid_inputs, args.valid_abs, valid_summary_dir,
        method=args.method, summary_length=args.summary_length)

    test_summary_dir = os.path.join(
        args.eval_dir, "summaries", args.method, "test")

    if not os.path.exists(test_summary_dir):
        os.makedirs(test_summary_dir)

    test_rouge_df, test_rouge_conf = evaluate_method(
        args.test_inputs, args.test_abs, test_summary_dir,
        method=args.method, summary_length=args.summary_length)
    
    d = {"valid": {"rouge": valid_rouge_df.to_dict(),
                   "conf": valid_rouge_conf.to_dict(),},
         "test": {"rouge": test_rouge_df.to_dict(),
                  "conf": test_rouge_conf.to_dict()}}
    eval_path = os.path.join(
        args.eval_dir, "eval-results", 
        "{}.eval-results.json".format(args.method))
    with open(eval_path, "w") as fp:
        fp.write(json.dumps(d))

if __name__ == "__main__":
    main()
