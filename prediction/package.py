import argparse
import pathlib
import shutil
import datetime
from submission_namer import rename_submission
import uuid
import json
import tarfile


renaming_script_path = "./material_create_submission_filename-v0.1.3.py"
SYSTEM_NAME = "en_src_emb_sim-hl2-kw-umdnmt2_1-morph3_0-asr5_0"

now = datetime.datetime.now().isoformat().split(".")[0] + "Z"
IMAGE_TEMPLATE = "{team}.{system}.{query}.{doc}.png"
JSON_TEMPLATE = "{team}.{system}.{query}.{doc}.json"


def validate_results(results_paths, summary_dir):
    for results_path in results_path:
        with open(results_path, "r", encoding="utf8") as fp:
            query_id, query_domain_strin = fp.readline().strip().split()
            for line in fp: 
                doc_id, _ = line.strip().split()
                image_path = summary_dir / query_id / "{}.png".format(doc_id)
                json_path = summary_dir / query_id / "{}.json".format(doc_id)
                #if not image_path.exists():
                #"Missing 

def parse_results_tsv(path):
    results = []
    data = {"results": results}
    with open(path, "r", encoding="utf8") as fp:
        query_id, query_domain_string = fp.readline().strip().split("\t")
        data["query_id"] = query_id
        query_string, domain_string = query_domain_string.rsplit(":", 1)
        data["query_string"] = query_string
        data["domain_string"] = domain_string
        for line in fp:
            doc_id, score = line.strip().split("\t")
            results.append({"doc_id": doc_id, "score": score})
    return data



def create_query_tar(input_dir, target_dir, clir_results, run_name):
    target_dir.mkdir(parents=True, exist_ok=True)
    e2e_results = []
    for result in clir_results["results"]:
        doc_id = result["doc_id"]
        score = result["score"]
        
        # Copy image path.
        source_image_path = input_dir / "{}.png".format(doc_id)
        target_image_path = target_dir / IMAGE_TEMPLATE.format(
            team="SCRIPTS", 
            system=SYSTEM_NAME, 
            query=clir_results["query_id"], 
            doc=doc_id)
        target_image_path.write_bytes(source_image_path.read_bytes())
    
        source_json_path = input_dir / "{}.json".format(doc_id)
        target_json_path = target_dir / JSON_TEMPLATE.format(
            team="SCRIPTS", 
            system=SYSTEM_NAME, 
            query=clir_results["query_id"], 
            doc=doc_id)
        summary_data = json.loads(
            source_json_path.read_bytes(), encoding="utf8")

        summary_data["team_id"] = "SCRIPTS"
        summary_data["sys_label"] = SYSTEM_NAME
        summary_data["uuid"] = str(uuid.uuid4())
        summary_data["query_id"] = clir_results["query_id"]
        summary_data["document_id"] = doc_id
        summary_data["run_name"] = run_name
        summary_data["run_date_time"] = now
        summary_data["image_uri"] = str(target_image_path.name)

        target_json_path.write_text(
            json.dumps(summary_data), encoding="utf8")
        e2e_results.append("\t".join([doc_id, score, target_json_path.name]))

    e2e_query_tsv = target_dir / "s-{query_id}.tsv".format(**clir_results)
    e2e_query_text = "\n".join([
        "{query_id}\t{query_string}:{domain_string}".format(**clir_results),
        *e2e_results])
    e2e_query_tsv.write_text(e2e_query_text, encoding="utf8")

#    tar_path = target_dir.parent / "{query_id}.tgz".format(**clir_results)
#    with tarfile.open(tar_path, "w:gz") as tar:
#        for path in target_dir.glob("*"):
#            arc_path = pathlib.Path(clir_results["query_id"]) / path.name
#            tar.add(
#                str(path), 
#                arcname=str(arc_path))   
#    shutil.rmtree(str(target_dir))
#    return tar_path.name

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query-folder", type=str, required=True)
    parser.add_argument("--results-folder", type=str, required=True)
    parser.add_argument("--summary-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--package-dir", type=str, required=True)
    parser.add_argument("--exp-path", type=str, required=True)
    
    args = parser.parse_args()

    summary_dir = pathlib.Path(args.summary_dir)
    
    package_dir = pathlib.Path(args.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    results_dir = pathlib.Path(args.results_folder)

    #results_paths = [x for x in results_dir.glob("*.tsv")]
    #results_paths.sort()

    #validate_results(results_paths, summary_dir)
    

    query_dir_paths = []
    for summary_query_dir in summary_dir.glob("query*"):
        query_id = summary_query_dir.name
        
        clir_results_tsv = results_dir / "q-{}.tsv".format(query_id)
        clir_results = parse_results_tsv(clir_results_tsv)
        target_dir = package_dir / query_id
        create_query_tar(
            summary_query_dir, target_dir, clir_results, args.run_name)
        query_dir_paths.append(target_dir)
    
    tmp_tar_name = package_dir / "{}.tgz".format(str(uuid.uuid4()))
    
    with tarfile.open(tmp_tar_name, "w:gz") as tar:
        for query_dir_path in query_dir_paths:
            
            #canon_tar_path = package_dir / local_dir_path
            tar.add(query_dir_path, arcname=query_dir_path.name)
            shutil.rmtree(str(query_dir_path))
            #canon_tar_path.unlink()

    pipeline_data = json.loads(
        pathlib.Path(args.exp_path).read_bytes(), encoding="utf8")
    rename_submission(pipeline_data, str(tmp_tar_name), str(package_dir), renaming_script_path)
    tmp_tar_name.unlink()

if __name__ == "__main__":
    main()
