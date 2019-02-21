from subprocess import check_output
import json


NON_BREAKING = set(
    [",", ";", "-RRB-", ".", "?", "''", "'m", "!", "...", "'s", "n't", 
     "'ll", "'all", "'re", "'d", "'ve"])

TWOSIDED = set(["--", ":", "-", "/" ])

SPACE_CONSUMING = set(["``", "`", "(", "$"])

def get_morph(document, client_path, port_num, lang):
    # document is a list of list of token strings: 
    # i.e. [["A", "sentence", "."], ["Another", "sentence", "."]]

    doc_morph = []

    for sentence in document:
        raw_output = check_output(
            ["java", "-jar", client_path, port_num, lang, " ".join(sentence)])
        data = json.loads(raw_output)
        doc_morph.append([t for s in data for t in s])

    for s_idx, sentence in enumerate(doc_morph):
        for t_idx, tok in enumerate(sentence):
            if tok["tense"] == "PPAST":
                tok["tense"] = "PAST"
            if tok["word"] == "-LRB-":
                tok["word"] = "("
            elif tok["word"] == "-RRB-":
                tok["word"] = ")"
            if tok["pos"] == "PNC":
                tok["wc"] = 0
            else:
                tok["wc"] = 1
            if t_idx == 0:
                tok["sstart"] = True
            else:
                if sentence[t_idx - 1]["consume_space"]:
                    tok["wc"] = 0
                tok["sstart"] = False

#            if s_idx > 0 and t_idx == 0 \
#                    and doc_morp[s_idx-1][-1]["pos"] != "PNC":
#                # If the start of new s 
#                tok["nl"] = True
#            else:
#                tok["nl"] = False
            if t_idx + 1 < len(sentence) \
                    and sentence[t_idx + 1]["word"] in NON_BREAKING:
                tok["consume_space"] = True
            elif tok["word"] in SPACE_CONSUMING:
                tok["consume_space"] = True
            else:
                tok["consume_space"] = False
    return doc_morph

def get_query_morph(query_tokens, client_path, port_num, lang):
    # query_tokens is a list of strings, e.g. ["Example", "query"]

    raw_output = check_output(
        ["java", "-jar", client_path, port_num, lang, " ".join(query_tokens)])
    data = json.loads(raw_output)
    m = [t for s in data for t in s]
    for t in m:
        if t["tense"] == "PPAST":
            t["tense"] = "PAST"
    return m

def get_morph2(document, client_path, port_num, lang):
    # document is a list of list of token strings: 
    # i.e. [["A", "sentence", "."], ["Another", "sentence", "."]]

    doc_morph = []

    raw_output = check_output(
            ["java", "-jar", client_path, port_num, lang, " ".join(document)])
    data = json.loads(raw_output)
#    for sentence in data:
#        doc_morph.append([t for s in data for t in s])

    for s_idx, sentence in enumerate(data):
        for t_idx, tok in enumerate(sentence):
            if tok["tense"] == "PPAST":
                tok["tense"] = "PAST"
            if tok["word"] == "-LRB-":
                tok["word"] = "("
            elif tok["word"] == "-RRB-":
                tok["word"] = ")"
            if tok["pos"] == "PNC":
                tok["wc"] = 0
            else:
                tok["wc"] = 1
            if t_idx == 0:
                tok["sstart"] = True
            else:
                if sentence[t_idx - 1]["consume_space"]:
                    tok["wc"] = 0
                tok["sstart"] = False

#            if s_idx > 0 and t_idx == 0 \
#                    and doc_morp[s_idx-1][-1]["pos"] != "PNC":
#                # If the start of new s 
#                tok["nl"] = True
#            else:
#                tok["nl"] = False

            if tok["word"] in TWOSIDED:
                tok["consume_space"] = True
                if t_idx >= 1:
                    sentence[t_idx - 1]["consume_space"] = True 
            elif t_idx + 1 < len(sentence) \
                    and sentence[t_idx + 1]["word"] in NON_BREAKING:
                tok["consume_space"] = True
            elif tok["word"] in SPACE_CONSUMING:
                tok["consume_space"] = True
            else:
                tok["consume_space"] = False
            doc_morph.append(tok)
    return doc_morph


