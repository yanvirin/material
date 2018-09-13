# Jessica Ouyang
# summary_specific_instructions.py


GREEN = '#39FF14'
PURPLE = '#DA70D6'

def parse_query_helper(query):
    constraint, morphological, example = None, None, None
    conceptual = False

    if '"' in query:
        query = query.replace('"', '')
    
    if '[' in query:
        start, end = query.find('['), query.find(']')
        constraint = query[start+5:end]
        query = query[:start] + query[end+1:]

    if query.endswith('+'):
        conceptual = True
        query = query[:-1]
        
    elif '<' in query:
        morphological = query[query.find('<')+1 : query.find('>')]
        query = query.replace('<', '').replace('>', '')
        
    elif 'EXAMPLE_OF' in query:
        example = query[query.find('(')+1 : query.find(')')]
        query = query.replace('EXAMPLE_OF(', '').replace(')', '')
        
    return (query.lower(), constraint, conceptual, example, morphological)

def parse_query(query):
    query = query.split(',')
    return [parse_query_helper(subquery.strip()) for subquery in query]


def get_instructions(raw_query, exact_match_list, not_found_list):
    all_outputs = []

    queries = parse_query(raw_query)
    for i in range(len(queries)):
        query, constraint, conceptual, example, morphological = queries[i]
        
        term_word = 'word'
        if conceptual:
            term_word = 'topic'
        else:
            term_word = 'phrase'
        
        if i == 0:
            color, color_text = GREEN, 'green'
        else:
            color, color_text = PURPLE, 'purple'

        constrained_query = '%s (%s)' % (query, constraint) if constraint else query
        line1 = 'The %s <b><font color="%s">%s</font></b> appears in <b><font color="%s">%s</font></b> in the top portion of the summary above.' % (term_word, color, constrained_query, color, color_text)

        line2 = []
        exact_matches = [word for word in exact_match_list if word.lower() in query.lower()]
        num_matches = len(exact_matches)
        if num_matches > 0:
            match_word = 'word'
            if num_matches > 1:
                match_word = match_word + 's'
            line2.append('We found an exact match for the %s <b><font color="%s">%s</font></b> in the document. If it appears in the summary, it is highlighted in yellow.' % (match_word, color, ' '.join(exact_matches)))

        not_found = [word for word in not_found_list if word.lower() in query.lower()]
        num_missing = len(not_found)
        some_missing = num_missing > 0
        if some_missing:
            missing_word = 'word'
            if num_missing > 1:
                missing_word = missing_word + 's'
            line2.append('We did not find an exact match for some words. These words are shown at the bottom of the summary.') #% (missing_word, color, ' '.join(not_found)))
            
        line2 = ' '.join(line2)

        line3 = ''
        if not some_missing:
            if conceptual:                
                line3 = 'Check the summary to make sure it mentions the topic of <b><font color="%s">%s</font></b>.' % (color, constrained_query)
                if constraint:
                    line3 = line3 + ' Make sure to look for the specific sense of <b><font color="%s">%s</font></b> given by <b><font color="%s">%s</font></b>.' % (color, query, color, constraint)
            elif example:
                line3 = 'Check the summary to make sure it mentions an example of <b><font color="%s">%s</font></b>.' % (color, constrained_query)
                if constraint:
                    line3 = line3 + ' Make sure to look for the specific sense of <b><font color="%s">%s</font></b> given by <b><font color="%s">%s</font></b>.' % (color, query, color, constraint)
            elif constraint:
                line3 = 'Check the summary to make sure the specific sense of <b><font color="%s">%s</font></b> given by <b><font color="%s">%s</font></b> appears.' % (color, query, color, constraint)

        else: # some query terms not found
            if conceptual:
                line3 = 'Read the summary and look for words or phrases that are related to the topic of <b><font color="%s">%s</font></b>.' % (color, constrained_query)
                
            elif example:
                line3 = 'Read the summary and look for words or phrases that are examples of <b><font color="%s">%s</font></b>.' % (color, constrained_query)

            else:
                line3 = 'Read the summary and look for words or phrases that mean the same thing as %s <b><font color="%s">%s</font></b>.' % (term_word, color, constrained_query)

            if morphological:
                line3 = line3[:-1] + ' AND have exactly the same number (singular vs. plural, for nouns) or tense (for verbs).'

            if constraint:
                line3 = line3 + ' Make sure to look for the specific sense of <b><font color="%s">%s</font></b> given by <b><font color="%s">%s</font></b>.' % (color, query, color, constraint)
            
        all_outputs.append('\n'.join([line1, line2, line3]))
    return all_outputs

domain_instructions = {
    "Government-And-Politics": "If the document seems like it would belong in the Politics section of a newspaper (not necessarily an English newspaper), it probably discusses the topic of Government and Politics. A detailed definition of the topic, with examples, is shown below.",
    "Lifestyle": "If the document seems like it would belong in the Culture, Fashion, Food, Home and Garden, or Travel section of a newspaper (not necessarily an English newspaper), it probably discusses the topic of Lifestyle. A detailed definition of the topic, with examples, is shown below.",
    "Business-And-Commerce": "If the document seems like it would belong in the Business, Economy, or Markets section of a newspaper (not necessarily an English newspaper), it probably discusses the topic of Business and Commerce. A detailed definition of the topic, with examples, is shown below.",
    "Law-And-Order": "If the document seems like it would belong in the Law section of a newspaper (not necessarily an English newspaper), it probably discusses the topic of Law and Order. A detailed definition of the topic, with examples, is shown below.",
    "Physical-And-Mental-Health": "If the document seems like it would belong in the Health or Wellness section of a newspaper (not necessarily an English newspaper), it probably discusses the topic of Physical and Mental Health. A detailed definition of the topic, with examples, is shown below.",
    "Military": "A detailed definition of the topic of Military, with examples, is shown below.",
    "Sports": "If the document seems like it would belong in the Sports section of a newspaper (not necessarily an English newspaper), it probably discusses the topic of Sports. A detailed definition of the topic, with examples, is shown below."}

def get_domain_instructions(domain_string):
    return domain_instructions[domain_string]
