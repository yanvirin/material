import numpy as np
import morphology_client
import json
from .matching_util import tokens2string, sim
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english') + \
    ["also", '"', "'", "``", "''", ",", ".", ";", ":", "?", "/", "\\", "!",
     "(", ")", "~", "`", "[", "]", "{", "}", "-", "_", "+", "=", "@", "#",
     "$", "%", "^", "&", "*", "|"])



def score_petra_sentences(sentences, domain_id, embeddings):
    dm = {"GOV": "government", "BUS": "business", "LAW": "law",
          "REL": "religion", "MIL": "military"}
    domain_emb = embeddings[dm[domain_id]]

    all_scores = []
    for sent in sentences:
        scores = []
        for token in sent:
            token = token.lower()
            if token in STOPWORDS:
                continue
            scores.append(sim(embeddings[token], domain_emb))
        if len(scores) == 0:
            all_scores.append(0.)
        else:
            all_scores.append(np.mean(scores))
    return all_scores

def petra_domain(result, system_context, domain_id, morph_path, port, 
                 query_data, colors, budget):

    domain_map = {"Government-And-Politics": "GOV", 
                  'Business-And-Commerce': "BUS", 
                  'Law-And-Order': "LAW", 
                  'Military': "MIL", 
                  'Religion': "REL"}
    doc_id = result["doc_id"]
    with open(result["domain_id_path"], "r") as fp:
        headers = fp.readline().strip().split(",")[1:]
        headers = [domain_map[h] for h in headers]
        hidx = headers.index(domain_id)
        
        sent_ids = []
        for line in fp:
            if not line.startswith(doc_id):
                continue
            items = line.strip().split(",")
            headers = items[1:]
            sid = int(items[0].split("-")[-1])
            dlabel = int(headers[hidx])
            sent_ids.append((sid, dlabel))
            
    if len(sent_ids) != len(result["document_tokens"]):
        raise Exception("Bad domain id/document alignment!")
            
    pos_sent_ids = [x for x in sent_ids if x[1] == 1]
  
    if len(pos_sent_ids) == 0:
        pos_sent_ids = [x for x in sent_ids 
                        if len(result["document_tokens"][x[0]]) > 5]

    sentences = [result["document_tokens"][x[0]] for x in pos_sent_ids
                 if len(result["document_tokens"][x[0]]) > 5]
    scores = score_petra_sentences(
        sentences, domain_id, system_context["english_embeddings"]["model"])

    tokens = []
    for idx in np.argsort(scores)[::-1]:
        tokens.extend(sentences[idx])
        if len(tokens) >= budget:
            break
    tokens = tokens[:budget]
    morph = morphology_client.get_morph2(
        tokens,
        morph_path,
        port,
        "ENG")
    for i, t in enumerate(morph):
        t["highlight"] = False
        t["nl"] = t["sstart"] and i > 0
    for q, c in zip(query_data, colors):
        highlight_excerpt(morph, query_data[q]["query_tokens"],
                         system_context["english_embeddings"]["model"],
                         c)

    if len(morph) > 0:
        morph[-1]["consume_space"] = True
        morph.append(
             {"word": "...", "highlight": False, "consume_space": False, 
             "nl": False})
    excerpt_string = tokens2string(morph)

    dm = {"GOV": "government", "BUS": "business", "LAW": "law",
          "REL": "religion", "MIL": "military"}
    return {"type": "domain",
            "tokens": morph,
            "excerpt_string": excerpt_string,
            "message": "DOMAIN RELEVANT ({}):".format(dm[domain_id]),
            "message_color": "yellow"}


    
     
        #for line in fp:
        #    if line.startswith(doc_id):
        #        print(line.strip())


def domain(result, system_context, domain_id, morph_path, port, query_data,
           colors, budget):
    domain_maps = {"GOV": "government", "MIL": "military", "REL": "religion",
                   "BUS": "business", "LAW": "law"} 
    domain_id = domain_maps[domain_id]
    with open(result["domain_id_path"], "r") as fp:
        domain_probs = json.loads(fp.read())
   
    if len(domain_probs) != len(result["document_tokens"]):
        raise Exception("Bad domain id/document alignment!")
    scores = [score[domain_id] for score in domain_probs]
    for i, toks in enumerate(result["document_tokens"]):
        if len(toks) < 10:
            scores[i] = 0.

    tokens = []
    for idx in np.argsort(scores)[::-1]:
        tokens.extend(result['document_tokens'][idx])
        if len(tokens) >= budget:
            break
    tokens = tokens[:budget]
    morph = morphology_client.get_morph2(
        tokens,
        morph_path,
        port,
        "ENG")
    for i, t in enumerate(morph):
        t["highlight"] = False
        t["nl"] = t["sstart"] and i > 0
    for q, c in zip(query_data, colors):
        highlight_excerpt(morph, query_data[q]["query_tokens"],
                         system_context["english_embeddings"]["model"],
                         c)

    morph[-1]["consume_space"] = True
    morph.append(
            {"word": "...", "highlight": False, "consume_space": False, 
             "nl": False})
    excerpt_string = tokens2string(morph)
    return {"location": idx, "type": "domain",
            "tokens": morph,
            "excerpt_string": excerpt_string,
            "message": "DOMAIN RELEVANT ({}):".format(domain_id),
            "message_color": "yellow"}

def highlight_excerpt(excerpt_tokens, query_tokens, embeddings, color):

    for query_token in query_tokens:
        query_token = query_token.lower()
        matches = []
        for token in excerpt_tokens:
            if token["word"].lower() in STOPWORDS:
                match = 0.
            elif query_token == token["word"].lower():
                match = 1.
                token["color"] = color
                token["highlight"] = True
            elif query_token in embeddings \
                    and token["word"].lower() in embeddings:
                match = max(0, sim(embeddings[query_token], 
                                   embeddings[token["word"].lower()]))
            else:
                match = 0.
            matches.append(match)
        for idx in np.argsort(matches)[::-1][:3]:
            if matches[idx] > 0:
                excerpt_tokens[idx]["color"] = color
