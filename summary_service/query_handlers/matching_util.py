import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


STOPWORDS = ["have", "has", "had", "be", "to", "will", "became", "is", "are", 
             "was", "were", "'", '"', "``", "''", "(", ")", "the", "a", "an"]

def sim(x, y):
    if x is None or np.all(x == 0):
        return 0.
    elif y is None or np.all(y == 0):
        return 0.
    else:
        return 1 - cosine_distance(x, y)

def get_embedding(token, embeddings):
    found_embeddings = []
    word = token["word"].lower() 
    if word in embeddings:
        found_embeddings.append(embeddings[word])
    stem = token["stem"].lower()
    if stem in embeddings:
        found_embeddings.append(embeddings[stem])
    if len(found_embeddings) == 0:
        return None
    elif len(found_embeddings) == 1:
        return found_embeddings[0]
    else:
        return sum(found_embeddings) / 2

def get_embedding_matches(query_tokens, document_tokens, embeddings):
    all_matches = []
    for query_token in query_tokens:
        matches = []
        query_embedding = get_embedding(query_token, embeddings)
        
        if query_embedding is None:
            all_matches.append(np.array([0.] * len(document_tokens)))
            continue
        
        for token in document_tokens:
            token_embedding = get_embedding(token, embeddings)    
                     
            if token_embedding is None:            
                soft_match = 0.
            elif token["pos"] in ["OTHER", "PNC"]:
                soft_match = 0.
            elif token["word"].lower() in STOPWORDS:
                soft_match = 0.
            else:
                soft_match = max(0, sim(token_embedding, query_embedding))
            matches.append(soft_match)

        all_matches.append(np.array(matches))
    return sum(all_matches)

def get_phrase_embedding_matches(query_tokens, document_tokens, embeddings):
    all_matches = []
    for query_token in query_tokens:
        matches = []
        query_embedding = get_embedding(query_token, embeddings)
        
        for token in document_tokens:
            token_embedding = get_embedding(token, embeddings)    
            
            if token["word"].lower() == query_token["word"].lower():
                soft_match = 1.
            elif token["stem"].lower() == query_token["stem"].lower():
                soft_match = 1.
            elif token_embedding is None:            
                soft_match = 0.
            else:
                soft_match = max(0, sim(token_embedding, query_embedding))
            matches.append(soft_match)

        all_matches.append(np.array(matches))
    return sum(all_matches)

def compute_bounds(doc_morph_flat, budget):
    start_stops = []
    left_win = budget // 2
    right_win = budget - left_win
    for i, token in enumerate(doc_morph_flat):
        #if token["pos"] in ["PNC", "OTHER"]:
          #  start_stops.append([i,i+1])
          #  continue
        
        left_wc = doc_morph_flat[i]["wc"]
        if i == 0:
            j = 0
        else:
            for j in range(i-1, -1, -1):
                left_wc += doc_morph_flat[j]["wc"]
                if left_wc == left_win:
                    break
        left_start = j
        right_wc = 0
        for j in range(i+1, len(doc_morph_flat)):
            right_wc += doc_morph_flat[j]["wc"]
            if right_wc + left_wc == budget:
                break
        right_stop = j

        start_stops.append([left_start,right_stop])
    return np.array(start_stops)

def compute_density(matches, bounds, context_scores):

    densities = []    
    for idx, (l, r) in enumerate(bounds):
        match_count = matches[l:r].sum()
        ctx_score = context_scores[l:r].sum()
        densities.append(match_count * ctx_score)
    return np.array(densities)        


def tokens2string(tokens):
    buffer = ""
    for token in tokens:
        if token["nl"]:
            buffer += "\n"
        buffer += token["word"]
        if not token["consume_space"]:
            buffer += " "
    return buffer.strip()
