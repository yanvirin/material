import re

def resolve_translation_path(doc_id, language, system_context):
    root_dir = system_context["nist_data"] / language / \
        "IARPA_MATERIAL_BASE-{}".format(language)
    
    for part in ["DEV", "ANALYSIS1", "ANALYSIS2", "EVAL1", "EVAL2", "EVAL3"]:
        text_path = root_dir / part / "text" / "mt_store" / \
            system_context["translation"]["text"] / "{}.txt".format(
            doc_id)
        if text_path.exists():
            return part, "text", text_path
        audio_path = root_dir / part / "audio" / "mt_store" / \
            system_context["translation"]["audio"] / "{}.txt".format(
            doc_id)
        if audio_path.exists():
            return part, "audio", audio_path
        raise Exception("translation failed: text path %s, audio path %s do not exist" % 
          (text_path, audio_path))

def load_clir_results(clir_results_path, system_context):
    results = []
    with open(clir_results_path, "r") as fp:
        header = fp.readline().strip()
        for line in fp:
            doc_id, relevance_score = line.strip().split()
            relevance_score = float(relevance_score)
            match = re.match(r"MATERIAL_BASE-(\d[A-Z])_\d+", doc_id)
            language = match.groups()[0]
            part, source, translation_path = resolve_translation_path(
                doc_id, language, system_context)

            morph_path = system_context["nist_data"] / language / \
                "IARPA_MATERIAL_BASE-{}".format(language) / \
                part / source / "morphology_store" / \
                system_context["morphology"][source] / \
                "{}.txt".format(doc_id)

            
            if not morph_path.exists():
              raise Exception("morph path does not exist: %s" % morph_path)

            result = {"doc_id": doc_id, "language": language, 
                      "dataset": part, "source": source, 
                      "translation_path": translation_path,
                      "morphology_path": morph_path}
            results.append(result)
    return results
