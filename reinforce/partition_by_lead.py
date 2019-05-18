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

def find_references(ref_summary_dir, doc_id):
    paths = []
    for fn in os.listdir(ref_summary_dir):
        ref_id = fn.split(".")[0]
        if ref_id == doc_id:
            paths.append(os.path.join(ref_summary_dir, fn))
    return paths

def partition_inputs(inputs_path, abstracts, inputs_out, output_dir, summary_length=100):

    ids = []
    rouge_config_paths = []
    for f in os.listdir(inputs_path):
        with open("%s/%s"%(inputs_path,f), "r") as inp_fp:
            example = json.load(inp_fp)
            sys_summary_text = make_lead(example, limit=summary_length)
            ref_paths = find_references(abstracts, example["id"])
            sys_path = os.path.join(output_dir, "{}.summary".format(example["id"]))
            with open(sys_path, "w") as out_fp: out_fp.write(sys_summary_text)
            rouge_config_paths.append([sys_path, ref_paths])
            ids.append(example["id"])
    with rouge_papier.util.TempFileManager() as manager:
        config_text = rouge_papier.util.make_simple_config_text(rouge_config_paths)
        config_path = manager.create_temp_file(config_text)
        df, conf = rouge_papier.compute_rouge(
                config_path, max_ngram=2, lcs=True, 
                remove_stopwords=False,
                length=summary_length, return_conf=True)
        df.index = ids + ["average"]
        scored_ids = sorted(df.to_dict()["rouge-1"].items(),key=lambda x: x[1])
    
    os.system("mkdir -p %s" % inputs_out)
    os.system("mkdir -p %s/%s" % (inputs_out, "inputs1"))
    os.system("mkdir -p %s/%s" % (inputs_out, "inputs2"))
    os.system("mkdir -p %s/%s" % (inputs_out, "inputs3"))
    os.system("mkdir -p %s/%s" % (inputs_out, "inputs4"))
    
    idx = int(len(scored_ids)/4)
    for id,score in scored_ids[:idx]:
        if id != "average":
          os.system("cp %s/%s.json %s/inputs1/" % (inputs_path,id,inputs_out))
    for id,score in scored_ids[idx:2*idx]:
        if id != "average":
          os.system("cp %s/%s.json %s/inputs2/" % (inputs_path,id,inputs_out))     
    for id,score in scored_ids[2*idx:3*idx]:
        if id != "average":
          os.system("cp %s/%s.json %s/inputs3/" % (inputs_path,id,inputs_out))
    for id,score in scored_ids[3*idx:4*idx]:
        if id != "average":
          os.system("cp %s/%s.json %s/inputs4/" % (inputs_path,id,inputs_out))    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--abstracts", type=str, required=True)
    parser.add_argument("--inputs-out", type=str, required=True)
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--summary-length", type=int, default=100)

    args = parser.parse_args()
    partition_inputs(args.inputs, args.abstracts, args.inputs_out, args.eval_dir, summary_length=args.summary_length)

if __name__ == "__main__":
    main()
