import json
from nltk import word_tokenize
import re
import lda
import sentence_ranker
import numpy as np
import string
import image_generator
import summary_instructions
import logging
import run_compressor
import traceback
import os

def read_morphology(path):
    data = []
    with open(path, "r", encoding="utf8") as fp:
        for line in fp:
            datum = json.loads(line)
            if len(datum):
                data.append([x for x in map(lambda x: x["word"], datum[0])])
            else:
                data.append([])
    return data

def read_translation(path):
    with open(path, "r", encoding="utf8") as fp:
        return [word_tokenize(line) for line in fp if len(line.strip())]

def get_query_content(query, keep_constraints=True):
    if "," in query:
        q1, q2 = query.split(",")
        return [get_query_content(q1, keep_constraints=keep_constraints)[0], 
                get_query_content(q2, keep_constraints=keep_constraints)[0]]
    if keep_constraints:
         query = re.sub(r"\[(hyp|evf|syn):(.*?)\]", r" \2 ", query)
    else:
         query = re.sub(r"\[(hyp|evf|syn):(.*?)\]", r" ", query)

    if '<' in query:
        query = query.replace('<', '').replace('>', '')
    if '"' in query:
        query = query.replace('"', '')
    if "'" in query:
        query = query.replace("'", ' ')
    query = query.replace("+", " ")
    query = re.sub(r"EXAMPLE_OF\((.*?)\)", r"\1", query)

    return [query.split()]

def get_topics(document, query_content, system_context):
    if not system_context["topic_model"]["use_topic_model"]:
        return None, 0
    if system_context["topic_model"]["model"] is None:
        raise Exception("Topic model not loaded!")

    model = system_context["topic_model"]["model"]

    return lda.get_topics(
        model, document, query_content,
        system_context["topic_model"]["max_topic_words"],
        always_highlight=True)

def get_topic_header(doc_id, system_context):
    topic_header_path = "%s/%s" % (system_context["topic_headers"], doc_id + ".topics")
    topic_header = []
    if os.path.isfile(topic_header_path):
      with open(topic_header_path) as tf:
        topic_header = tf.readlines()[0].split(" ")
    return topic_header, summary_length(topic_header)

def compress(sentences, constraints, system_context):
    assert(len(sentences) == len(constraints))
    assert("model" in system_context["compressor"])
    compressor = system_context["compressor"]["model"]
    print("inputs to compressor: %s, %s" % (str(sentences), str(constraints)))
    try:
      output = run_compressor.compress(compressor, sentences, constraints)
    except Exception as e:
      print("ERROR in compressing ^^^^" + str(e))
      raise Exception("failed to compress")
    return output

def is_punctuation(word):
    for char in word:
        if not char in string.punctuation:
            return False
    return True

def summary_length(sen):
    return len([token for token in sen if not is_punctuation(token)])

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

def get_translated_query(query_data):
    trans_data = list(filter(lambda x: "indri" in x and x["indri"].startswith("#combine ( ") and 
                        x["indri"].endswith(")") and x["type"]=="UMDPSQPhraseBased", query_data["queries"]))
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

def summarize_example(example, system_context):
    print(example["query"])
    summary_word_budget = system_context["summary_length"]
    query_content = get_query_content(example["query"])

    query_content_no_constraints = get_query_content(
        example["query"], keep_constraints=False)
    extract_summary = [word_tokenize(line) for line in example["summary_lines"]
                       if len(line.strip())]
    topics, topic_word_count = get_topics(
        extract_summary, query_content, system_context)
    summary_word_budget -= topic_word_count

    query_matches = []
    query_misses = []

    doc_words = set([w.lower() for s in extract_summary for w in s])
    for subquery in query_content_no_constraints:
        for term in subquery:
            if term.lower() in doc_words:
                query_matches.append(term)
            else:
                query_misses.append(term)
 

    component1_hl_weights = image_generator.calculate_highlight_weights(
        query_content_no_constraints[0], extract_summary, 
        system_context["english_embeddings"]["model"],
        system_context["english_stopwords"]["model"],
        threshold_topk=2)

    if len(query_content) == 2:
        component2_hl_weights = image_generator.calculate_highlight_weights(
            query_content_no_constraints[1], extract_summary, 
            system_context["english_embeddings"]["model"],
            system_context["english_stopwords"]["model"],
            threshold_topk=2)

        for c1_hl, c2_hl in zip(component1_hl_weights, component2_hl_weights):
            for i, (w1, w2) in enumerate(zip(c1_hl, c2_hl)):
                if w1 > w2:
                    c2_hl[i] = 0.0
                else:
                    c1_hl[i] = 0.0

    else:
        component2_hl_weights = None 

    image_generator.generate_image(
        example["output_path"], extract_summary, topics=topics, 
        highlight_weights1=component1_hl_weights,
        highlight_weights2=component2_hl_weights,
        missing_keywords=query_misses)

    
    matches_json_path = example["output_path"].replace(".png", ".matches.json")
    
    with open(matches_json_path, "w") as fp:
        fp.write(json.dumps({"query_misses": query_misses, "query_matches": query_matches}))

def summarize_query_part(query_content, summary_word_budget, doc_translation, doc_morphology,
                         bad_alignment, system_context, query_data):
    sentence_rankings = []

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
        trans_query = get_translated_query(query_data)
        ranking = sentence_ranker.query_embedding_similarity(
            doc_morphology, trans_query, emb)
        sentence_rankings.append(ranking)
    
    if system_context["sentence_rankers"]["lexical-expansion-translation"]:
        qestring = None
        for query in query_data["queries"]:
          if "expanded_words" in query:
            qestring = query["expanded_words"]
        query_expansion = [ws.split(":") for ws in qestring.split(";")]
        query_expansion = [(ws[0], float(ws[1])) if len(ws) == 2 else (ws[0], 1.)
                           for ws in query_expansion]
        query_words = query_data["english"]["words"]
        ranking = sentence_ranker.query_lexical_similarity(
            doc_translation, query_content, query_expansion)
        sentence_rankings.append(ranking)

    ranking = merge_rankings(sentence_rankings)
    best_sentence_indices = np.argsort(ranking)
    extract_summary = create_extract_summary(
      best_sentence_indices, doc_translation, summary_word_budget)

    if system_context["compressor"]["use_compressor"]:
      constraints = [[item for sublist in query_content for item in sublist]]*len(extract_summary)
      return compress(extract_summary, constraints, system_context)
    return extract_summary

def summarize(query_content, domain_id_sen, summary_word_budget, doc_translation, doc_morphology, 
              bad_alignment, query_data, system_context):
    extract_summary = [domain_id_sen] if domain_id_sen else []
    summary_word_budget -= summary_length(domain_id_sen)
    if system_context["separated"] and len(query_content) > 1:
      for query_part in query_content:
        partial_summary = summarize_query_part([query_part], summary_word_budget, doc_translation, doc_morphology,
                        bad_alignment, system_context, query_data)
        if len(partial_summary) > 0 and partial_summary[0] not in extract_summary:
          extract_summary.append(partial_summary[0])
          summary_word_budget -= summary_length(partial_summary[0])

    summary = summarize_query_part(query_content, summary_word_budget*5, doc_translation, doc_morphology,
                        bad_alignment, system_context, query_data)
    for sentence in summary:
      if sentence not in extract_summary and summary_word_budget > 0:
        temp_sen = []
        for token in sentence:
          temp_sen.append(token)
          if not is_punctuation(token): summary_word_budget -= 1
          if summary_word_budget == 0: break
          assert(summary_word_budget > 0)
        extract_summary.append(temp_sen)
    return extract_summary

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

def summarize_query_result(result, query_data, system_context):
    query_id = query_data["parsed_query"][0]["info"]["queryid"]
    domain_id = query_data["domain"]["domain_id"]
    doc_id = result["doc_id"]

    domain_id_scores = system_context["domain_id"][domain_id] if "domain_id" in system_context else None

    logging.info(" summarizing {} - {}".format(query_id, doc_id))

    doc_morphology = read_morphology(result["morphology_path"])
    doc_translation = read_translation(result["translation_path"])
    summary_word_budget = system_context["summary_length"]

    bad_alignment = len(doc_translation) != len(doc_morphology)

    if bad_alignment:
        logging.error(
            " Translation/Morphology length not equal!\n morph={}\n trans={})".format(
                result["morphology_path"],
                result["translation_path"]))

    logging.info(" query string: {}".format(query_data["IARPA_query"]))
    query_content = get_query_content(query_data["IARPA_query"])
    logging.info(" query_content: {}".format(str(query_content) ))
    
    '''if system_context["topic_model"]["use_topic_model"]:
      if system_context["topic_model"]["use_topic_cache"]:
        cache = system_context["topic_model"]["topic_cache"]
        key = "{}-{}".format(query_id, doc_id)
        if key in cache:
            topics, topic_word_count = cache[key]
        else:
            topics, topic_word_count = get_topics(
                doc_translation, query_content, system_context)
            cache[key] = (topics, topic_word_count)
      else:
        topics, topic_word_count = get_topics(
            doc_translation, query_content, system_context)'''

    topic_header, topic_word_count = get_topic_header(doc_id, system_context)
   
    summary_word_budget -= (topic_word_count+1)

    doc_words = set([w.lower() for s in doc_translation for w in s])
    query_matches = []
    query_misses = []

    query_content_no_constraints = get_query_content(
        query_data["IARPA_query"], keep_constraints=False)
    for subquery in query_content_no_constraints:
        for term in subquery:
            if term.lower() in doc_words:
                query_matches.append(term)
            else:
                query_misses.append(term)
 
    summary_word_budget = summary_word_budget - len(query_misses) - 3

    domain_id_sen = doc_translation[np.argmax(domain_id_scores)] if \
                    domain_id_scores and len(domain_id_scores) == len(doc_translation) else []

    #doc_translation_path = str(result["translation_path"])
    #doc_type = "[audio]" if "/audio/" in doc_translation_path else "[text]" if "/text/" in doc_translation_path else "[unknown]"
    extract_summary = summarize(query_content, domain_id_sen, summary_word_budget, doc_translation, doc_morphology,
              bad_alignment, query_data, system_context)

    if system_context["split"]:
      extract_summary = fix_summary(extract_summary, query_content, domain_id_sen, summary_word_budget, 
                                    doc_translation, doc_morphology, bad_alignment, 
                                    query_data, system_context)

    # if len(extract_summary) > 0: extract_summary.append(doc_type)

    component1_hl_weights = image_generator.calculate_highlight_weights(
        query_content_no_constraints[0], extract_summary, 
        system_context["english_embeddings"]["model"],
        system_context["english_stopwords"]["model"],
        threshold_topk=2)

    if len(query_content) == 2:
        component2_hl_weights = image_generator.calculate_highlight_weights(
            query_content_no_constraints[1], extract_summary, 
            system_context["english_embeddings"]["model"],
            system_context["english_stopwords"]["model"],
            threshold_topk=2)

        for c1_hl, c2_hl in zip(component1_hl_weights, component2_hl_weights):
            for i, (w1, w2) in enumerate(zip(c1_hl, c2_hl)):
                if w1 > w2:
                    c2_hl[i] = 0.0
                else:
                    c1_hl[i] = 0.0

    else:
        component2_hl_weights = None 

                   
    image_path = system_context["summary_dir"] / query_id / \
        "{}.png".format(result["doc_id"])

    image_generator.generate_image(
        image_path, extract_summary, topics=topic_header, 
        highlight_weights1=component1_hl_weights,
        highlight_weights2=component2_hl_weights,
        missing_keywords=query_misses) 

    instructions = summary_instructions.get_instructions(
        query_data["IARPA_query"], query_matches, query_misses)

    instructions = {"component_{}".format(i): instr 
                    for i, instr in enumerate(instructions, 1)} 
    
    word_list = []
    '''if topics:
        word_list.extend(["QUERY", "TOPICS"])
        for topic in topics:
            word_list.extend(topic["query"])
            word_list.extend(topic["topic_words"])'''
    word_list.extend(topic_header)

    word_list.append("SUMMARY")
    word_list.extend([w for s in extract_summary for w in s 
                      if not is_punctuation(w)])
    if len(query_misses):
        word_list.extend(["WORDS", "NOT", "FOUND"] + query_misses)

    if len(word_list) > 100:
        logging.warn(" {}/{} Summary word list > 100 words: {}".format(
            query_id, doc_id, len(word_list)))

    meta = {"word_list": word_list,
            "instructions": instructions}

    meta_path = system_context["summary_dir"] / query_id / \
        "{}.json".format(result["doc_id"])

    with open(meta_path, "w", encoding="utf8") as fp:
        fp.write(json.dumps(meta))

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
