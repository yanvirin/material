import json
import numpy as np
import sentence_ranker
import string
from .matching_util import sim, tokens2string 
import re
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english') + ["also"])


def generic(result, query_data, system_context, budget, color):
    message = "MOST RELEVANT SENTENCES ({}):".format(
        query_data["query_string"])
    msg_len = len(message.split(" "))

    sentence_rankings = []
    doc_translation = result["document_tokens"]
    doc_morphology = read_source(result["morphology_path"])
    bad_alignment = len(doc_translation) != len(doc_morphology)
    query_content = list(query_data["query_tokens"])
    if query_data["constraint_lc_tokens"] is not None:
        query_content.extend(query_data["constraint_lc_tokens"])
    query_content = [query_content] 

    if system_context["sentence_rankers"]["qa"]:
        for question_word in system_context["qa_question_words"]:
            ranking = sentence_ranker.query_qa_similarity(
                doc_translation, query_content, question_word)
            sentence_rankings.append(ranking)
    
    if system_context["sentence_rankers"]["translation"] or bad_alignment:
        ranking = sentence_ranker.query_embedding_similarity(
            doc_translation, query_content,
            system_context["english_embeddings"]["model"])
        sentence_rankings.append(ranking)

    if system_context["sentence_rankers"]["source"] and not bad_alignment:
        emb = system_context["source_embeddings"]["model"]
        trans_query = get_translated_query(query_data["original_query_data"])
        ranking = sentence_ranker.query_embedding_similarity(
            doc_morphology, trans_query, emb)
        sentence_rankings.append(ranking)
    
    if system_context["sentence_rankers"]["lexical-expansion-translation"]:
        qestring = None
        for query in query_data["query_expansions"]:
          if "expanded_words" in query:
            qestring = query["expanded_words"]
        query_expansion = [ws.split(":") for ws in qestring.split(";")]
        query_expansion = [(ws[0], float(ws[1])) if len(ws) == 2 else (ws[0], 1.)
                           for ws in query_expansion]
        ranking = sentence_ranker.query_lexical_similarity(
            doc_translation, query_content, query_expansion)
        sentence_rankings.append(ranking)

    ranking = merge_rankings(sentence_rankings)
    best_sentence_indices = np.argsort(ranking)
    extract_summary = create_extract_summary(
      best_sentence_indices, doc_translation, budget - msg_len)

#    if system_context["split"]:
#      print("I'll fix you!")
#      extract_summary = fix_summary(extract_summary, query_content, 
#                                    domain_id_sen, budget, 
#                                    doc_translation, doc_morphology, 
#                                    bad_alignment, 
#                                    query_data, system_context)



    excerpt_tokens = create_excerpt(extract_summary)
    highlight_excerpt(excerpt_tokens, query_data["query_tokens"], 
                      system_context["english_embeddings"]["model"], color)
    excerpt_string = tokens2string(excerpt_tokens)

    return {"tokens": excerpt_tokens, "type": "generic",
            "message": message,
            "excerpt_string": excerpt_string,
            "message_color": "yellow"}


#STOPWORDS = set(["a", "an", "the", "i", "we", "you", "us", "me", "she", "it",
#                 "his", "her", "is", "was", "are", "be", "were", "have",
                 

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


NONBREAKING = set([",", ";", ")", ".", "?", "''", ":", "'m", "!", "...", "'s",
                   "n't", "'ll", "'all", "'re", "'d", "'ve"])
CONSUME_SPACE = set(["``", "`", "(", "$"])

def create_excerpt(extract_summary):
    summary_tokens = []
    for s in extract_summary:
        for i, t in enumerate(s):
            if i + 1 < len(s) and s[i+1] in NONBREAKING:
                con_sp = True
            elif t in CONSUME_SPACE:
                con_sp = True
            else:
                con_sp = False

            summary_tokens.append(
                {"word": t, "highlight": False, "consume_space": con_sp,
                 "nl": i == 0})
    summary_tokens[-1]["consume_space"] = True
    summary_tokens.append(
            {"word": "...", "highlight": False, "consume_space": False, 
             "nl": False})
    return summary_tokens

def merge_rankings(rankings):
    """
    Aggregates multiple rankings to produce a single overall ranking of a list
    of items.

    rankings: a list of lists, each sublist is a ranking (0 is best)

    For example, let's say we are ranking 4 items A, B, C, and D and we have
    4 sets of a rankings. The input rankings might look like this:
    rankings = [[0, 1, 2, 3],
                [3, 0, 1, 2],
                [3, 1, 0, 2],
                [3, 1, 2, 0]]

    where: 
    [0, 1, 2, 3] expresses a preference for A, B, C, D with A the best;
    [3, 0, 1, 2] expresses a preference for B, C, D, A with B the best;
    [3, 1, 0, 2] expresses a preference for C, B, D, A with C the best;
    [3, 1, 2, 0] expresses a preference for D, B, C, A with D the best.

    agg_ranking = borda_count_rank_merge(rankings) 

    print(agg_ranking)
    [3 0 1 2] which corresponds to the ranking B, C, D, A  
    """

    # trivial case
    if len(rankings) == 1: return rankings[0]

    all_points = [0 for _ in rankings[0]]
    max_rank = len(rankings[0]) - 1
    for ranking in rankings:
        for i, rank in enumerate(ranking):
            points = max_rank - rank
            all_points[i] += points
    order = np.argsort(all_points)[::-1].tolist()
    agg_rank = [order.index(i) for i in range(max_rank + 1)]
    return agg_rank

def create_extract_summary(indices, translation, summary_word_budget):

    word_count = 0
    summary_lines = []
    if summary_word_budget == 0: return summary_lines

    for index in indices:
        summary_line = []
        for token in translation[index]:
            if is_punctuation(token):
                summary_line.append(token)
            else:
                summary_line.append(token)
                word_count += 1
                if word_count == summary_word_budget:
                    break
        summary_lines.append(summary_line)
        if word_count == summary_word_budget:
            break

    return summary_lines

def is_punctuation(word):
    for char in word:
        if not char in string.punctuation:
            return False
    return True

def read_source(path):
    data = []
    with open(path, "r", encoding="utf8") as fp:
        for line in fp:
            datum = json.loads(line)
            if len(datum):
                data.append([x for x in map(lambda x: x["word"], datum[0])])
            else:
                data.append([])
    return data

def fix_summary(extract_summary, query_content, domain_id_sen, summary_word_budget, doc_translation, doc_morphology,
              bad_alignment, query_data, system_context):
    bad_sens = [(i,s) for (i,s) in enumerate(extract_summary) if summary_length(s) > summary_word_budget/2]
    if len(bad_sens) == 0: return extract_summary
    (bad_sum_idx, bad_sen) = bad_sens[0]
    assert(len(bad_sens)==1)
    # try to fix the bad sentence
    query_terms = set([i.lower() for s in query_content for i in s])
    punct_idx = -1
    for i in reversed(range(len(bad_sen))):
      if is_punctuation(bad_sen[i]): punct_idx = i
      if bad_sen[i] in query_terms or i < len(bad_sen)/2: break
    if punct_idx < 0: return extract_summary
    new_sen = bad_sen[:punct_idx]
    doc_translation = [new_sen if s == bad_sen else s for s in doc_translation]
    new_summary = summarize(query_content, domain_id_sen, summary_word_budget, doc_translation, doc_morphology,
              bad_alignment, query_data, system_context)
    new_sens = [(i,s) for (i,s) in enumerate(new_summary) if s == new_sen]
    if len(new_sens) == 0 or new_sens[0][0] != bad_sum_idx: return extract_summary
    #if len(new_summary) == len(extract_summary): return extract_summary
    print("summary was fixed with splitting.")
    return new_summary

def get_translated_query(query_data):
    trans_data = list(filter(lambda x: "indri" in x and x["indri"].startswith("#combine ( ") and 
                        x["indri"].endswith(")") and x["type"]=="UMDPSQPhraseBasedGVCC", query_data["queries"]))
    assert(len(trans_data)>0)
    indri_str = trans_data[0]["indri"][11:-1]

    results = []
    for item in re.findall(r"(\w+)|#wsyn\((.*?)\)", indri_str):
        if item[0] == '':
            # found a weighted synset of (prob word) pairs.
            prob_words = item[1].split(" ")
            result = []
            for i in range(0, len(prob_words), 2):
                prob, word = prob_words[i:i+2]
                prob = float(prob)
                result.append((prob, word))
            results.append(result)
        else:
            # There was no translation -- use english query word and hope!
            results.append([(1.0, item[0])])

    return [term for result in results for score, term in result]


