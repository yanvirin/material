import re
from nltk import word_tokenize
from collections import OrderedDict
from pprint import pprint
from morphology_client import get_query_morph


def parse_query(iarpa_query_string, query_data, morph_path, morph_port):
    result = OrderedDict()
    if "," in iarpa_query_string:
        items = iarpa_query_string.split(",")
        for i, item in enumerate(items, 1):
            comp = "component{}".format(i)
            result[comp] = _parse_component(item, query_data, 
                                            morph_path, morph_port) 
            result[comp]["query_expansions"] = query_data["queries"]
            
    else:
        result["component1"] = _parse_component(iarpa_query_string, query_data,
                                                morph_path, morph_port)
        result["component1"]["query_expansions"] = query_data["queries"]
    return result

def _parse_component(query_string, qd, morph_path, morph_port):
    query_data = {}
    query_data["original_query_string"] = query_string

    query_string = re.sub(r"EXAMPLE_OF\((.*?)\)", r"\1", query_string)
    constraint_match = re.search(r"\[(hyp|evf|syn):(.*?)\]", query_string)
    if constraint_match:
        cons_type = constraint_match.groups()[0]
        cons_string = constraint_match.groups()[1]
        cons_tokens = word_tokenize(cons_string)
        query_data["constraint_type"] = cons_type
        query_data["constraint_string"] = cons_string
        query_data["constraint_lc_tokens"] = [t.lower() for t in cons_tokens]
        query_data["constraint_tokens"] = cons_tokens
        query_data["query_string"] = re.sub(r"\[(hyp|evf|syn):(.*?)\]", 
                                            r"", query_string)
        query_data["constraint_token_morphology"] = get_query_morph(
            cons_tokens, 
            morph_path,
            morph_port, "EN")
    else:
        query_data["constraint_type"] = None
        query_data["constraint_string"] = None
        query_data["constraint_lc_tokens"] = None 
        query_data["constraint_tokens"] = None
        query_data["constraint_token_morphology"] = None

    query_string = re.sub(r"\[(hyp|evf|syn):(.*?)\]", "", query_string)
    concept_match = re.search("^.*?\+$", query_string)
    if concept_match:
        query_string = query_string[:-1]
        query_data["conceptual"] = True
    else:
        query_data["conceptual"] = False

    morph_match = re.search(r"<(.*?)>", query_string)
    if morph_match:
        query_data["token_morphology"] = resolve_morph_query(
            qd["parsed_query"][0]["info"]["morph_word"])

        query_data["morphology_constraint"] = morph_match.groups()[0].split()
        query_data["morphology_context_tokens"] = None
        if query_string[0] == '"' and query_string[-1] == '"':
            query_data["is_phrase"] = True
            ctx_tokens = word_tokenize(
                re.sub(r"<.*?>", r" ", query_string[1:-1]))
            query_data["morphology_context_tokens"] = ctx_tokens
            query_data["query_tokens"] = word_tokenize(
                re.sub(r"<|>", r" ", query_string[1:-1]))
            query_data["query_string"] = re.sub(r"<|>", r" ", 
                                                query_string[1:-1])
        else:
            query_data["is_phrase"] = False
            query_data["query_tokens"] = word_tokenize(
                re.sub(r"<|>", r" ", query_string))
            query_data["query_string"] = re.sub(r"<|>", r" ", query_string)
            
        query_data["query_token_morphology"] = get_query_morph(
            query_data["query_tokens"], 
            morph_path,
            morph_port, "EN")

    else:    
        query_data["morphology_constraint"] = None
        if query_string[0] == '"' and query_string[-1] == '"':
            query_data["is_phrase"] = True
            query_tokens = word_tokenize(query_string[1:-1])
            query_data["query_tokens"] = query_tokens
            query_data["query_token_morphology"] = get_query_morph(
                query_tokens, 
                morph_path,
                morph_port,
                "EN")
            query_data["query_string"] = query_string[1:-1]
        else:
            query_data["is_phrase"] = False
            query_tokens = word_tokenize(query_string)
            query_data["query_tokens"] = query_tokens
            query_data["query_token_morphology"] = get_query_morph(
                query_tokens, 
                morph_path,
                morph_port, 
                "EN")
            query_data["query_string"] = query_string

    query_data["query_type"] = categorize(query_data) 
 
    return query_data

def categorize(query_data):

    if query_data["morphology_constraint"]:
        return "morphological_lexical"

    if not query_data["is_phrase"] and not query_data["conceptual"] \
            and query_data["morphology_constraint"] is None \
            and query_data["constraint_type"] is None \
            and len(query_data["query_tokens"]) == 1:
        return "unconstrained_simple_lexical"

    if not query_data["is_phrase"] and not query_data["conceptual"] \
            and query_data["morphology_constraint"] is None \
            and query_data["constraint_type"] is not None \
            and len(query_data["query_tokens"]) == 1:
        return "constrained_simple_lexical"

    if (query_data["is_phrase"] or len(query_data["query_tokens"]) > 1) \
            and not query_data["conceptual"] \
            and query_data["morphology_constraint"] is None \
            and query_data["constraint_type"] is None:
        return "unconstrained_simple_phrase"
    if (query_data["is_phrase"] or len(query_data["query_tokens"]) > 1) \
            and not query_data["conceptual"] \
            and query_data["morphology_constraint"] is None \
            and query_data["constraint_type"] is not None:
        return "constrained_simple_phrase"

    #print(query_data["original_query_string"])
   
    return "generic"

def resolve_morph_query(query_morph):
    if isinstance(query_morph, dict):
        return query_morph
    if isinstance(query_morph, list):
        query_morph = [x for x in query_morph if x["pos"] != "OTHER"]
        if len(query_morph) == 1:
            return query_morph[0]
        
        verbs = [x for x in query_morph if x["pos"] != "VB"]
        if len(verbs) > 1:
            return verbs[0]
        return query_morph[0]

