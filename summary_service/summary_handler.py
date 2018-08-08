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

def get_query_content(query):
    if "," in query:
        q1, q2 = query.split(",")
        return [get_query_content(q1)[0], get_query_content(q2)[0]]
    query = re.sub(r"\[(hyp|evf|syn):(.*?)\]", r" \2 ", query)
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

def is_punctuation(word):
    for char in word:
        if not char in string.punctuation:
            return False
    return True

def create_extract_summary(indices, translation, summary_word_budget):

    word_count = 0
    summary_lines = []
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
    trans_data = query_data["translations"][3]
    assert trans_data["Indri_query"].startswith("#combine(") \
        and trans_data["Indri_query"].endswith(")")
    indri_str = trans_data["Indri_query"][9:-1]

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
    extract_summary = [word_tokenize(line) for line in example["summary_lines"]
                       if len(line.strip())]
    topics, topic_word_count = get_topics(
        extract_summary, query_content, system_context)
    summary_word_budget -= topic_word_count

    component1_hl_weights = image_generator.calculate_highlight_weights(
        query_content[0], extract_summary, 
        system_context["english_embeddings"]["model"],
        system_context["english_stopwords"]["model"],
        threshold_topk=2)

    if len(query_content) == 2:
        component2_hl_weights = image_generator.calculate_highlight_weights(
            query_content[1], extract_summary, 
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
        highlight_weights2=component2_hl_weights)

def summarize_query_result(result, query_data, system_context):
    query_id = query_data["parsed_query"][0]["info"]["queryid"]
    doc_id = result["doc_id"]

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
    
    if system_context["topic_model"]["use_topic_model"] and \
            system_context["topic_model"]["use_topic_cache"]:
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
            doc_translation, query_content, system_context)
   
    summary_word_budget -= (topic_word_count + 3)
    sentence_rankings = []

    if system_context["sentence_rankers"]["translation"] or bad_alignment:
        ranking = sentence_ranker.query_embedding_similarity(
            doc_translation, query_content, 
            system_context["english_embeddings"]["model"])
        sentence_rankings.append(ranking) 

    if system_context["sentence_rankers"]["source"] and not bad_alignment:
        if result["language"] == "1A":
            emb = system_context["swahili_embeddings"]["model"]
        elif result["language"] == "1B":
            emb = system_context["tagalog_embeddings"]["model"]
        else:
            raise Exception("Bad language! No embeddings for {}".format(
                result["language"]))

        trans_query = get_translated_query(query_data)
        ranking = sentence_ranker.query_embedding_similarity(
            doc_morphology, trans_query, 
            emb)
        sentence_rankings.append(ranking) 
    if system_context["sentence_rankers"]["lexical-expansion-translation"]:
        
        qestring = query_data["english"]["expanded"][1]["expanded_words"]
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
    #("\n".join([" ".join(l) for l in extract_summary]))    

    component1_hl_weights = image_generator.calculate_highlight_weights(
        query_content[0], extract_summary, 
        system_context["english_embeddings"]["model"],
        system_context["english_stopwords"]["model"],
        threshold_topk=2)

    if len(query_content) == 2:
        component2_hl_weights = image_generator.calculate_highlight_weights(
            query_content[1], extract_summary, 
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

    doc_words = set([w.lower() for s in doc_translation for w in s])
    query_matches = []
    query_misses = []
    for subquery in query_content:
        for term in subquery:
            if term.lower() in doc_words:
                query_matches.append(term)
            else:
                query_misses.append(term)
                    
    image_path = system_context["summary_dir"] / query_id / \
        "{}.png".format(result["doc_id"])

    image_generator.generate_image(
        image_path, extract_summary, topics=topics, 
        highlight_weights1=component1_hl_weights,
        highlight_weights2=component2_hl_weights) 

    instructions = summary_instructions.get_instructions(
        query_data["IARPA_query"], query_matches, query_misses)

    instructions = {"component_{}".format(i): instr 
                    for i, instr in enumerate(instructions, 1)} 
    instructions["domain"] = summary_instructions.get_domain_instructions(
        query_data["domain"]["desc"])


    word_list = []
    if topics:
        word_list.extend(["QUERY", "TOPICS"])
        for topic in topics:
            word_list.extend(topic["query"])
            word_list.extend(topic["topic_words"])

    word_list.append("SUMMARY")
    word_list.extend([w for s in extract_summary for w in s 
                      if not is_punctuation(w)])

    if len(word_list) > 100:
        logging.warn(" {}/{} Summary word list > 100 words.".format(
            query_id, doc_id))

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
