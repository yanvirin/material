import numpy as np
from .matching_util import (get_embedding_matches, compute_bounds, 
                            compute_density, tokens2string)


def constrained_simple_lexical(result, query_data, system_context, 
                               budget, color):

    query_term = query_data["query_tokens"][0]
    doc_morph_flat = result["document_morphology_flat"]
    query_morph = query_data["query_token_morphology"]
    soft_matches = get_embedding_matches(
        query_morph + query_data["constraint_token_morphology"], 
        doc_morph_flat, system_context["english_embeddings"]["model"])

    exact_match = check_exact_match(
        query_data, doc_morph_flat, soft_matches, budget, color)
    if exact_match:
        return exact_match
    
    stem_match = check_stem_match(
        query_data, doc_morph_flat, soft_matches, budget, color)
    if stem_match:
        return stem_match

    soft_match = check_soft_match(
        query_data, doc_morph_flat, soft_matches, budget, color)
    if soft_match:
        return soft_match

    return None

def check_exact_match(query_data, doc_morph_flat, ctx_scores, budget, color):
    qt_lc = query_data["query_token_morphology"][0]["word"].lower()
    matches = np.array([qt_lc == token["word"].lower() 
                        for token in doc_morph_flat])
    if not np.any(matches):
       return None

    msg_tmp = "EXACT MATCH ({}), CHECK THAT WORD SENSE MATCHES \"{}\":"
    message = msg_tmp.format(query_data["query_tokens"][0],
                             query_data["constraint_string"])
    msg_len = len(message.split(" "))
    bounds = compute_bounds(doc_morph_flat, budget - msg_len)
    density = compute_density(matches, bounds, ctx_scores)
    match = np.argmax(density)
    l, r = bounds[match]
    match_tokens = doc_morph_flat[l:r]
    matches = matches[l:r]
    
    excerpt_tokens = []
    excerpt_tokens.append({"word": "...", "wc": 0, "consume_space": True,
                           "nl": False, "highlight": False})
    for token, match in zip(match_tokens, matches):
        t = {"word": token["word"],
             "wc": token["wc"],
             "consume_space": token["consume_space"],
             "nl": token.get("nl", False)}
        if match:
            t["highlight"] = True
            t["color"] = color
        else:
            t["highlight"] = False 
        excerpt_tokens.append(t)
    
    excerpt_tokens[-1]["consume_space"] = True
    excerpt_tokens.append({"word": "...", "wc": 0, "consume_space": False,
                           "nl": False, "highlight": False})
    
    excerpt_string = tokens2string(excerpt_tokens)

    return {"location": match, "type": "exact",
            "tokens": excerpt_tokens,
            "excerpt_string": excerpt_string,
            "message": message,
            "message_color": "chartreuse"}

def check_stem_match(query_data, doc_morph_flat, ctx_scores, budget, color):
    qt_lc = query_data["query_token_morphology"][0]["stem"].lower()
    matches = np.array([qt_lc == token["stem"].lower() 
                        for token in doc_morph_flat])
    if not np.any(matches):
       return None

    msg_tmp = "CLOSE MATCH ({}), CHECK THAT WORD SENSE MATCHES \"{}\":"
    message = msg_tmp.format(query_data["query_tokens"][0],
                             query_data["constraint_string"])
    msg_len = len(message.split(" "))
    bounds = compute_bounds(doc_morph_flat, budget - msg_len)
    density = compute_density(matches, bounds, ctx_scores)
    match = np.argmax(density)
    l, r = bounds[match]
    match_tokens = doc_morph_flat[l:r]
    matches = matches[l:r]
    
    excerpt_tokens = []
    excerpt_tokens.append({"word": "...", "wc": 0, "consume_space": True,
                           "nl": False, "highlight": False})
    for token, match in zip(match_tokens, matches):
        t = {"word": token["word"],
             "wc": token["wc"],
             "consume_space": token["consume_space"],
             "nl": token.get("nl", False)}
        if match:
            t["highlight"] = True
            t["color"] = color
        else:
            t["highlight"] = False 
        excerpt_tokens.append(t)
    
    excerpt_tokens[-1]["consume_space"] = True
    excerpt_tokens.append({"word": "...", "wc": 0, "consume_space": False,
                           "nl": False, "highlight": False})
    
    excerpt_string = tokens2string(excerpt_tokens)

    return {"location": match, "type": "stem",
            "tokens": excerpt_tokens,
            "message": message,
            "excerpt_string": excerpt_string,
            "message_color": "yellow"}

def check_soft_match(query_data, doc_morph_flat, ctx_scores, budget, color):

    if np.all(ctx_scores <= 0.1):
       return None
    matches = np.array([0.] * len(doc_morph_flat))
    for idx in np.argsort(ctx_scores)[::-1][:3]:
        matches[idx] = 1.


    message = "({}) NOT FOUND, SHOWING MOST SIMILAR WORDS:".format(
        query_data["query_tokens"][0])
    msg_len = len(message.split(" "))
    bounds = compute_bounds(doc_morph_flat, budget - msg_len)
    density = compute_density(matches, bounds, ctx_scores)
    match = np.argmax(density)
    l, r = bounds[match]
    match_tokens = doc_morph_flat[l:r]
    matches = matches[l:r]
    
    excerpt_tokens = []
    excerpt_tokens.append({"word": "...", "wc": 0, "consume_space": True,
                           "nl": False, "highlight": False})
    for token, match in zip(match_tokens, matches):
        t = {"word": token["word"],
             "wc": token["wc"],
             "consume_space": token["consume_space"],
             "nl": token.get("nl", False)}
        if match:
            t["highlight"] = False
            t["color"] = color
        else:
            t["highlight"] = False 
        excerpt_tokens.append(t)
    
    excerpt_tokens[-1]["consume_space"] = True
    excerpt_tokens.append({"word": "...", "wc": 0, "consume_space": False,
                           "nl": False, "highlight": False})
    
    excerpt_string = tokens2string(excerpt_tokens)

    return {"location": match, "type": "soft",
            "tokens": excerpt_tokens,
            "message": message,
            "excerpt_string": excerpt_string,
            "message_color": "deeppink"}
