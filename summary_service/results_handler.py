import re

def resolve_translation_path(doc_id, language, system_context):
    root_dir = system_context["nist_data"] / language / \
        "IARPA_MATERIAL_OP1-{}".format(language)
    
    for part in ["DEV", "ANALYSIS", "ANALYSIS1", "ANALYSIS2", "EVAL1", "EVAL2", "EVAL3"]:
        text_path = root_dir / part / "text" / \
            system_context["translation"]["text"] / "{}.txt".format(
            doc_id)
        if text_path.exists():
            return part, "text", text_path
        audio_path = root_dir / part / "audio" / \
            system_context["translation"]["audio"] / "{}.txt".format(
            doc_id)
        if audio_path.exists():
            return part, "audio", audio_path

    raise Exception("translation failed: text path %s, audio path %s do not exist" % 
      (text_path, audio_path))

def resolve_domain_id_path(doc_id, language, system_context, part, source):
    root_dir = system_context["nist_data"] / language / \
        "IARPA_MATERIAL_OP1-{}".format(language)

    if system_context["domain_handler"] == "apoorv":
        path = root_dir / part / source / "domainIdentification_store" / \
            system_context["domain_id_path"][source] / "{}.json".format(doc_id)
    else:
        path = root_dir / part / source / "domainIdentification_store" / \
            system_context["domain_id_path"][source] / \
            "sentence-predictions.csv"
    return path

def load_clir_results(clir_results_path, system_context):
    results = []
    with open(clir_results_path, "r") as fp:
        for line in fp:
            doc_id, decision, relevance_score = line.strip().split()
            if decision == "N": continue
            relevance_score = float(relevance_score)
            match = re.match(r"MATERIAL_[0-9A-Z]+-(\d[A-Z])_\d+", doc_id)
            language = match.groups()[0]
            part, source, translation_path = resolve_translation_path(
                doc_id, language, system_context)
            
            if system_context["domain_handler"] != "none":
                domain_id_path = resolve_domain_id_path(
                    doc_id, language, system_context, part, source)
            else:
                domain_id_path = None

            morph_path = system_context["nist_data"] / language / \
                "IARPA_MATERIAL_OP1-{}".format(language) / \
                part / source / "morphology_store" / \
                system_context["morphology"][source] / \
                "{}.txt".format(doc_id)

            
            if not morph_path.exists():
              raise Exception("morph path does not exist: %s" % morph_path)

            result = {"doc_id": doc_id, "language": language, 
                      "dataset": part, "source": source, 
                      "translation_path": translation_path,
                      "domain_id_path": domain_id_path, 
                      "morphology_path": morph_path}
            results.append(result)
    return results
