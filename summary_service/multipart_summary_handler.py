import query_parser
import query_handlers as qh
import morphology_client
import logging
from image_generator2 import SummaryImage
import summary_instructions2
from nltk import word_tokenize
import json
import re


def read_translation(path):
    with open(path, "r", encoding="utf8") as fp:
        return [word_tokenize(re.sub(r"<oov>", " ", line, flags=re.I))
                for line in fp if len(line.strip())]

extsum_handlers = {
    "unconstrained_simple_lexical": qh.unconstrained_simple_lexical,
    "constrained_simple_lexical": qh.constrained_simple_lexical,
    "unconstrained_simple_phrase": qh.unconstrained_simple_phrase,
    "constrained_simple_phrase": qh.constrained_simple_phrase,
    "morphological_lexical": qh.morphological,}

def compute_section_budgets(total_budget, query_data):
    if len(query_data) == 2:
        return {"component1": total_budget // 2,
                "component2": total_budget - total_budget // 2}
    else:
        return {"component1": total_budget}

def annotate_morphology(result, system_context, port):
    doc_morph = morphology_client.get_morph2(
        result["document_tokens_flat"], 
        system_context["morph_client_path"],
        port,
        "ENG")
#    doc_morph_flat = [t for s in doc_morph for t in s]
#    result["document_morphology"] = doc_morph
    result["document_morphology_flat"] = doc_morph
        

def get_topic_header(doc_id, system_context):
    if system_context["topic_headers"] is None:
        return None, 0
    
    path = system_context["topic_headers"] / doc_id
    if not path.exists():
        return None, 0
    header = path.read_text().strip()
    return header, len(header.split(" "))

def summarize_query_result(result, query_data, system_context,
        real_summary_length=None, round_two=False):

    orig_query_data = query_data
    query_id = query_data["parsed_query"][0]["info"]["queryid"]
    domain_id = query_data["domain"]["domain_id"]

    logging.info(
        "Reading translation from {}".format(result["translation_path"]))
    result["document_tokens"] = read_translation(result["translation_path"])
    result["document_tokens_flat"] = [t for s in result["document_tokens"]
                                      for t in s]
    
    logging.info("Getting translation morphology")
    annotate_morphology(result, system_context, 
                        system_context["morph_server_port"])

    doc_id = result["doc_id"]
    header, header_wc = get_topic_header(doc_id, system_context)
    oqs = query_data["IARPA_query"]
    query_data = query_parser.parse_query(
        query_data["IARPA_query"], query_data, 
        system_context["morph_client_path"],
        system_context["morph_server_port"])
    for qd in query_data.values():
        qd["original_query_data"] = orig_query_data
    #highlight_colors = ["#39FF14", "orchid"]

    light_green = "limegreen"
    dark_green = "darkgreen"
    highlight_colors = [light_green, "orchid"]

    if system_context["domain_handler"] == "apoorv":
        domain_id_extract = qh.domain(
            result, system_context, domain_id, 
            system_context["morph_client_path"],
            system_context["morph_server_port"],
            query_data, highlight_colors,
            30)
        for token in domain_id_extract["tokens"]:
            if token["highlight"] and token["color"] == light_green:
                token["color"] = dark_green
    elif system_context["domain_handler"] == "petra":
        domain_id_extract = qh.petra_domain(
            result, system_context, domain_id, 
            system_context["morph_client_path"],
            system_context["morph_server_port"],
            query_data, highlight_colors,
            30)
        for token in domain_id_extract["tokens"]:
            if token["highlight"] and token["color"] == light_green:
                token["color"] = dark_green
    else:
        domain_id_extract = None

    text_snippets = []
    image = SummaryImage()
#    image.append_message(oqs, "white")
#    image.append_message("", "white")
    if header is not None:
        text_snippets.append(header)
        image.append_message(header, "white")
        image.append_message("", "white")

    if real_summary_length is None:
        real_summary_length = system_context["summary_length"]
    summary_budget = real_summary_length - header_wc
    if domain_id_extract:
        summary_budget -= len(domain_id_extract["message"].split())
        summary_budget -= len(domain_id_extract["excerpt_string"].split())
    
    section_budgets = compute_section_budgets(summary_budget, query_data)
    summary_parts = []

    for (c, comp_data), color in zip(query_data.items(), highlight_colors):
        h = extsum_handlers.get(comp_data["query_type"], qh.generic) 
        summary = h(result, comp_data, system_context, 
                    section_budgets[c], color)
        if summary is None:
            summary = qh.generic(
                result, comp_data, system_context, 
                section_budgets[c], color)        
        if summary is not None:
            summary_parts.append(summary)

            for token in summary["tokens"]:
                if token["highlight"] and token["color"] == light_green:
                    token["color"] = dark_green
        else:
            raise Exception("NO SUMMARY HANDLER FOR QUERY TYPE!")
    for part in summary_parts:
        image.append_message("", "white")
        image.append_message(part["message"], part["message_color"])
        image.append_extract(part)
        text_snippets.append(part["message"])
        text_snippets.append(part["excerpt_string"])
        
    if domain_id_extract:
        image.append_message("", "white")
        image.append_message("", "white")
        image.append_message(domain_id_extract["message"], 
                             domain_id_extract["message_color"])
        image.append_extract(domain_id_extract)
        text_snippets.append(domain_id_extract["message"])
        text_snippets.append(domain_id_extract["excerpt_string"])


    image_path = system_context["summary_dir"] / query_id / \
        "{}.png".format(result["doc_id"])
    image.write(image_path)

    found_words = []
    missing_words = []
    doc_content = set([w for t in result["document_morphology_flat"] 
                       for w in [t["word"].lower(), t["stem"].lower()]])
    for query in query_data.values():
        for qw in query["query_token_morphology"]:
            if qw["word"].lower() in doc_content \
                    or qw["stem"].lower() in doc_content:
                found_words.append(qw["word"])
            else:
                missing_words.append(qw["word"])
    instructions = summary_instructions2.get_instructions(
        oqs, found_words, missing_words)

    wc = sum([len(s.split()) for s in text_snippets])
    if wc > system_context["summary_length"]:
        over = wc - system_context["summary_length"]
        real_summary_length = real_summary_length - over
        if round_two:
            logging.warn(" {}/{} Summary word list > 100 words: {}".format(
                query_id, doc_id, wc))
            logging.warn("Trying New length", real_summary_length)
        summarize_query_result(result, orig_query_data, system_context,
            real_summary_length=real_summary_length, round_two=False)
        return 
        #priont("\n".join(text_snippets))
        tc = 0
        for snippet in text_snippets:
            for subsnippet in snippet.split("\n"):
                for word in subsnippet.split():
                    print("{}_{}".format(tc, word), end=" ")
                    tc += 1
                print()
        print()
        tc = 0
        for token in summary_parts[0]["tokens"]:
            tc += token["wc"]
            print(tc, token["word"])

    instructions = {"component_{}".format(i): instr 
                    for i, instr in enumerate(instructions, 1)} 

    meta = {"content_list": text_snippets,
            "instructions": instructions}

    meta_path = system_context["summary_dir"] / query_id / \
        "{}.json".format(result["doc_id"])

    with open(meta_path, "w", encoding="utf8") as fp:
        fp.write(json.dumps(meta))


