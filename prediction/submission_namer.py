import os

def resolve_query_set(paths):
    query_sets = []
    for path in paths:
        if path.endswith(".tsv"):
            query_sets.append(os.path.basename(os.path.split(path)[0]))
        else:
            if path[-1] == "/":
                path = path[:-1]
            query_sets.append(os.path.basename(path))
    query_sets.sort()
    query_set = "".join(query_sets)
    valid_query_sets = set([
        "QUERY1",
        "QUERY2",
        "QUERY1QUERY2",
        "QUERY2QUERY3"])
    if query_set not in valid_query_sets:
        raise Exception("Bad query set: {}".format(query_set))
    return query_set


def resolve_datasets(paths):
    paths = [path[:-1] if path[-1] == "/" else path for path in paths]
    datasets = [os.path.basename(path) for path in paths]
    if not len(datasets) in [1, 2, 3]: 
        raise Exception("Bad dataset argument.")
    if len(datasets) == 1:
        if not datasets[0] in ["ANALYSIS1", "DEV", "EVAL1", "EVAL2", "EVAL3"]:
            return Exception("Bad dataset argument.")
        else:
            datasets = datasets[0]
    elif len(datasets) == 2:
        if "ANALYSIS1" in datasets and "ANALYSIS2" in datasets:
            datasets = "ANALYSIS1ANALYSIS2"
        elif "DEV" in datasets and "ANALYSIS1" in datasets:
            datasets = "DEVANALYSIS1"
        elif "EVAL1" in datasets and "EVAL2" in datasets:
            datasets = "EVAL1EVAL2"
        else:
            raise Exception("Bad dataset argument.")
    elif len(datasets) == 3:
        if "EVAL1" in datasets and "EVAL2" in datasets and "EVAL3" in datasets:
            datasets = "EVAL1EVAL2EVAL3"
        else:
            raise Exception("Bad dataset argument.")
    return datasets


def rename_submission(pipeline_data, tar_path, output_dir, 
                      rename_script_location):

    if pipeline_data["query_processor"]["target_language"] == "sw":
        lang = "1A"
    elif pipeline_data["query_processor"]["target_language"] == "tl":
        lang = "1B"
    elif pipeline_data["query_processor"]["target_language"] == "so":
        lang = "1S"
    else:
        raise Exception(
            "Bad language code! Must be 'sw', 'tl' or 'so' but found '{}'".format(
                pipeline_data["domain_modeling"]["target_language"]))
            
    queryset = resolve_query_set(
        pipeline_data["query_processor"]["query_list_path"])

    datasets = resolve_datasets(
        pipeline_data["data_collection"]["collections"])

    args = [
        "python", rename_script_location,
        "--team", "SCRIPTS",
        "--task", "E2E",
        "--sub_type", pipeline_data["submission_type"],
        "--lang", lang,
        "--set", queryset,
        "--dataset", datasets,
        "--submission_file", tar_path,
        "--outpath", output_dir,
    ]

    os.system(" ".join(args))
