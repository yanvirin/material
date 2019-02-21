import numpy as np
from .matching_util import (get_embedding_matches, get_embedding,
                            get_phrase_embedding_matches, compute_bounds, 
                            compute_density, tokens2string, sim)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english') + ["also"])

def check_bow_match(query_data, doc_morph_flat, ctx_scores, budget, color):

    query_bow = set()
    for token in query_data["query_token_morphology"]:
        query_bow.add(token["word"].lower())
        query_bow.add(token["stem"].lower())

    matches = np.array(
        [t["stem"].lower() in query_bow or t["word"].lower() in query_bow
         for t in doc_morph_flat])

    if not np.any(matches):
       return None

    msg_tmp = "CLOSE MATCH ({}), CHECK THAT PHRASE SENSE MATCHES \"{}\":"
    message = msg_tmp.format(
        " ".join(query_data["query_tokens"]),
        query_data["constraint_string"])
    msg_len = len(message.split(" "))
    bounds = compute_bounds(doc_morph_flat, budget - msg_len)
    density = compute_density(matches, bounds, ctx_scores)
    match = np.argmax(density)
    l, r = bounds[match]
    match_tokens = doc_morph_flat[l:r]
    matches = matches[l:r]
    
    excerpt_tokens = []
    excerpt_tokens.append({"word": "...", "stem": "...", "wc": 0, "pos": "PNC",
                           "consume_space": True,
                           "nl": False, "highlight": False})
    for token, match in zip(match_tokens, matches):
        t = {"word": token["word"],
             "wc": token["wc"],
             "pos": token["pos"],
             "stem": token["stem"],
             "consume_space": token["consume_space"],
             "nl": token.get("nl", False)}
        if match:
            t["highlight"] = True
            t["color"] = color
        else:
            t["highlight"] = False 
        excerpt_tokens.append(t)
    
    excerpt_tokens[-1]["consume_space"] = True
    excerpt_tokens.append({"word": "...", "stem": "...", "wc": 0, "pos": "PNC",
                           "consume_space": False,
                           "nl": False, "highlight": False})
    
    excerpt_string = tokens2string(excerpt_tokens)

    return {"location": match, "type": "bow",
            "tokens": excerpt_tokens,
            "message": message,
            "excerpt_string": excerpt_string,
            "message_color": "yellow"}

def highlight_excerpt(excerpt_tokens, query_tokens, embeddings, color):

    for query_token in query_tokens:
        query_emb = get_embedding(query_token, embeddings)
        query_word = query_token["word"].lower()
        query_stem = query_token["stem"].lower()
        matches = []
        for token in excerpt_tokens:
            token_emb = get_embedding(token, embeddings)
            if token["word"].lower() in STOPWORDS:
                match = 0.
            elif token["pos"] in ["PNC", "OTHER"]:
                match = 0.
            elif query_word == token["word"].lower() \
                    or query_stem == token["stem"].lower():
                match = 1.
                token["color"] = color
                token["highlight"] = True
            else:
                match = max(0, sim(query_emb, token_emb))
            matches.append(match)
        for idx in np.argsort(matches)[::-1][:3]:
            if matches[idx] > 0:
                excerpt_tokens[idx]["color"] = color


def constrained_simple_phrase(result, query_data, system_context, 
                                budget, color):

    doc_morph_flat = result["document_morphology_flat"]
    query_morph = query_data["query_token_morphology"]
    soft_matches = get_phrase_embedding_matches(
        query_morph + query_data["constraint_token_morphology"],
        doc_morph_flat, 
        system_context["english_embeddings"]["model"])

    bow_match = check_bow_match(
        query_data, doc_morph_flat, soft_matches, budget, color)
    if bow_match:
        highlight_excerpt(
            bow_match["tokens"],
            query_morph + query_data["constraint_token_morphology"], 
            system_context["english_embeddings"]["model"], color)
        return bow_match

    return None


