import numpy as np
from .matching_util import (get_embedding_matches, compute_bounds, sim,
                            compute_density, tokens2string, get_embedding)
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english') + ["also"])

def morphological(result, query_data, system_context, budget, color):

    doc_morph_flat = result["document_morphology_flat"]
    ctx_tokens = list(query_data["morphology_constraint"])
    if query_data["constraint_lc_tokens"] is not None:
        ctx_tokens.extend(query_data["constraint_lc_tokens"])
    if query_data["morphology_context_tokens"] is not None:
        ctx_tokens.extend(query_data["morphology_context_tokens"])

    #print(ctx_tokens)
    ctx_scores = get_context_match(
        doc_morph_flat, ctx_tokens, 
        system_context["english_embeddings"]["model"])

    exact_match = check_exact_matches(
        query_data, doc_morph_flat, ctx_scores, budget, color)
    if exact_match:
        #print("I am exact?!")
        query_morph = query_data["query_token_morphology"]
        if query_data["constraint_token_morphology"] is not None:
            query_morph.extend(query_data["constraint_token_morphology"])
        highlight_excerpt(exact_match["tokens"], query_morph, 
                          system_context["english_embeddings"]["model"], color)
        return exact_match

    soft_match = check_soft_matches(
        query_data, doc_morph_flat, ctx_scores, 
        system_context["english_embeddings"]["model"],
        budget, color)
    if soft_match:
        #print("I am not exact")
        query_morph = query_data["query_token_morphology"]
        if query_data["constraint_token_morphology"] is not None:
            query_morph.extend(query_data["constraint_token_morphology"])

        #print(query_morph)
        highlight_excerpt(soft_match["tokens"], query_morph, 
                          system_context["english_embeddings"]["model"], color)
        return soft_match




def check_exact_matches(query_data, doc_morph_flat, ctx_scores, budget, color):

    def check(t, q):
        return t["word"].lower() == q["word"].lower() \
            and t["pos"] == q["pos"] and t["number"] == q["number"] \
            and t["tense"] == q["tense"]
    qmorph = query_data["token_morphology"]
    matches = np.array([1. if check(t, qmorph) else 0.
                        for t in doc_morph_flat])


    if not np.any(matches == 1.):
        return None
    
    if query_data["morphology_context_tokens"] is None and \
            query_data["constraint_type"] is not None:
        msg = 'EXACT MATCH ({}), CHECK THAT WORD SENSE MATCHES "{}":'.format(
            query_data["token_morphology"]["word"],
            query_data["constraint_string"])
        msg_color = "chartreuse"
    elif query_data["morphology_context_tokens"] is not None or \
        query_data["constraint_type"] is not None:
        msg = "EXACT MATCH ({}), CHECK THAT MEANING IS CORRECT:".format(
            query_data["token_morphology"]["word"])
        msg_color = "yellow"
    else:
        msg = "EXACT MATCH ({}):".format(
            query_data["token_morphology"]["word"])
        msg_color = "chartreuse"
    msg_len = len(msg.split(" "))
    bounds = compute_bounds(doc_morph_flat, budget - msg_len)
    density = compute_density(matches, bounds, ctx_scores)
    match = np.argmax(density)
    l, r = bounds[match]
    match_tokens = doc_morph_flat[l:r]
    matches = matches[l:r]
    excerpt_tokens = []
    excerpt_tokens.append({"word": "...", "stem": "...", "pos": "PNC", 
                           "wc": 0, "consume_space": True,
                           "nl": False, "highlight": False})
    for token, match in zip(match_tokens, matches):
        t = {"word": token["word"],
             "wc": token["wc"],
             "stem": token["stem"],
             "pos": token["pos"],
             "consume_space": token["consume_space"],
             "nl": token.get("nl", False)}
        if match:
            t["highlight"] = True
            t["color"] = color
        else:
            t["highlight"] = False 
        excerpt_tokens.append(t)
    
    excerpt_tokens[-1]["consume_space"] = True
    excerpt_tokens.append({"word": "...", "stem": "...", "pos": "PNC",
                           "wc": 0, "consume_space": False,
                           "nl": False, "highlight": False})
    
    excerpt_string = tokens2string(excerpt_tokens)
    #print(len(excerpt_string.split()), excerpt_string)

    return {"location": match, "type": "exact",
            "tokens": excerpt_tokens,
            "excerpt_string": excerpt_string,
            "message": msg,
            "message_color": msg_color}

def check_soft_matches(query_data, doc_morph_flat, ctx_scores, embeddings, 
                       budget, color):

    query = query_data["token_morphology"]
    qemb = embeddings[query["word"].lower()]
    sims = []
    for t in doc_morph_flat:
        if t["word"].lower() in STOPWORDS:
            sims.append(0.)
        elif t["pos"] == query["pos"] and t["number"] == query["number"] \
                and t["tense"] == query["tense"]:
            sims.append(sim(qemb, embeddings[t["word"].lower()]))
        else:
            sims.append(0.)
        #if t["word"] in ["started", "to", "leave"]:
        #    print(sims[-1], t)
    matches = np.maximum(0., np.array(sims))

    qmorph = query_data["token_morphology"]

    if not np.any(matches > 0.):
        return None
        
    msg_tmp = "FOUND A SIMILAR WORD TO ({}), CHECK THAT MEANING IS CORRECT:"
    msg = msg_tmp.format(qmorph["word"])
    msg_color = "yellow"
    msg_len = len(msg.split(" "))
    bounds = compute_bounds(doc_morph_flat, budget - msg_len)
    density = compute_density(matches, bounds, ctx_scores)

   
    match = np.argmax(density)



#    matches[:] = 0
#    matches[match] = 1.

    l, r = bounds[match]
    match_tokens = doc_morph_flat[l:r]
    matches = matches[l:r]
    I = np.argsort(matches)[::-1][:3]
    matches[:] = 0
    for i in I:
        matches[i] = 1.

    #print(len(match_tokens))
    #print(matches.shape)
    excerpt_tokens = []
    excerpt_tokens.append({"word": "...", "stem": "...", "pos": "PNC", 
                           "wc": 0, "consume_space": True,
                           "nl": False, "highlight": False})
    for token, match in zip(match_tokens, matches):
        t = {"word": token["word"],
             "wc": token["wc"],
             "stem": token["stem"],
             "pos": token["pos"],
             "consume_space": token["consume_space"],
             "nl": token.get("nl", False)}
        if match:
            #print(t["word"], "MATCHING")
            t["highlight"] = False
            t["color"] = color
        else:
            t["highlight"] = False 
        excerpt_tokens.append(t)
    
    excerpt_tokens[-1]["consume_space"] = True
    excerpt_tokens.append({"word": "...", "stem": "...", "pos": "PNC",
                           "wc": 0, "consume_space": False,
                           "nl": False, "highlight": False})
    
    excerpt_string = tokens2string(excerpt_tokens)
    #print(len(excerpt_string.split()), excerpt_string)

    return {"location": match, "type": "soft",
            "tokens": excerpt_tokens,
            "excerpt_string": excerpt_string,
            "message": msg,
            "message_color": msg_color}



def get_context_match(tokens, queries, embeddings):

    qembs = []
    for q in queries:
        if q in embeddings:
            qembs.append(embeddings[q])

    if len(qembs) == 0:
        return np.array([0. for t in tokens])

    qemb = sum(qembs) / len(qembs)
    sims = []
    for t in tokens:
        if t["word"].lower() in embeddings:
            sims.append(sim(qemb, embeddings[t["word"].lower()]))
        else:
            sims.append(0.)
    sims = np.maximum(0., np.array(sims))
    return sims

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


