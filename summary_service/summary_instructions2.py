# Jessica Ouyang
# summary_specific_instructions.py

GREEN = '#39ff14'
GREEN = '#32CD32'
PURPLE = '#da70d6'

morph_alts = {
    'accepted': ['accepting', 'accept', 'accepts'],
    'accompanied': ['accompanying', 'accompany', 'accompanies'],
    'affected': ['affecting', 'affect', 'affects'],
    'allocated': ['allocating', 'allocate', 'allocates'],
    'announced': ['announcing', 'announce', 'announces'],
    'artists': ['artist'],
    'ate dinner': ['eating dinner', 'eat dinner', 'eats dinner'],
    'ate': ['eating', 'eat', 'eats'],
    'attacked': ['attacking', 'attack', 'attacks'],
    'authorities': ['authority'],
    'blessed': ['blessing', 'bless', 'blesses'],
    'bought': ['buying', 'buy', 'buys'],
    'brought': ['bringing', 'bring', 'brings'],
    'called': ['calling', 'call', 'calls'],
    'campaigned': ['campaigning', 'campaign', 'campaigns'],
    'citizens': ['citizen'],
    'competed': ['competing', 'compete', 'competes'],
    'completed': ['completing', 'complete', 'completes'],
    'copies': ['copy'],
    'corrected': ['correcting', 'correct', 'corrects'],
    'covered': ['covering', 'cover', 'covers'],
    'defeated': ['defeating', 'defeat', 'defeats'],
    'defended': ['defending', 'defend', 'defends'],
    'discrimiated': ['discriminating', 'discriminate', 'discriminates'],
    'dissatisfied': ['dissatisfying', 'dissatisfy', 'dissatisfies'],
    'doctors': ['doctor'],
    'doors': ['door'],
    'drank': ['drinking', 'drink', 'drinks'],
    'droughts': ['drought'],
    'drums': ['drum'],
    'elections': ['election'],
    'eliminated': ['eliminating', 'eliminate', 'eliminates'],
    'encouraged': ['encouraging', 'encourage', 'encourages'],
    'escaped': ['escaping', 'escape', 'escapes'],
    'escorted': ['escorting', 'escort', 'escorts'],
    'evacuated': ['evacuating', 'evacuate', 'evacuates'],
    'exhorted': ['exhorting', 'exhort', 'exhorts'],
    'failed': ['failing', 'fial', 'fails'],
    'farmers': ['farmer'],
    'fascinated': ['fascinating', 'fascinate', 'fascinates'],
    'fields': ['field'],
    'finished': ['finishing', 'finish', 'finishes'],
    'fixed': ['fixing', 'fix', 'fixes'],
    'gifts': ['gift'],
    'girls': ['girl'],
    'graduated': ['graduating', 'graduate', 'graduates'],
    'greeted': ['greeting', 'greet', 'greets'],
    'hats': ['hat'],
    'healed': ['healing', 'heal', 'heals'],
    'heard': ['hearing', 'hear', 'hears'],
    'hid': ['hiding', 'hide', 'hides'],
    'hotels': ['hotel'],
    'ideas': ['idea'],
    'improved': ['improving', 'improve', 'improves'],
    'injured': ['injuring', 'injure', 'injures'],
    'instruments': ['instrument'],
    'isolated': ['isolating', 'isolate', 'isolates'],
    'killed': ['killing', 'kill', 'kills'],
    'knew': ['knowing', 'know', 'knows'],
    'learned': ['learning', 'learn', 'learns'],
    'listened': ['listening', 'listen', 'listens'],
    'materials': ['material'],
    'medications': ['medication'],
    'mothers': ['mother'],
    'movements': ['movement'],
    'nationalities': ['nationality'],
    'offices': ['office'],
    'pardoned': ['pardoning', 'pardon', 'pardons'],
    'parents': ['parent'],
    'physicians': ['physician'],
    'pictures': ['picture'],
    'planned': ['planning', 'plan', 'plans'],
    'played': ['playing', 'play', 'plays'],
    'players': ['player'],
    'poems': ['poem'],
    'prayed': ['praying', 'pray', 'prays'],
    'prevented': ['preventing', 'prevent', 'prevents'],
    'promised': ['promising', 'promise', 'promises'],
    'promoted': ['promoting', 'promote', 'promotes'],
    'pushed': ['pushing', 'push', 'pushes'],
    'read': ['reading', 'reads'],
    'rejected': ['rejecting', 'reject', 'rejects'],
    'released': ['releasing', 'release', 'releases'],
    'rescued': ['rescuing', 'rescue', 'rescues'],
    'restaurants': ['restaurant'],
    'rivers': ['river'],
    'saluted': ['saluting', 'salute', 'salutes'],
    'saved': ['saving', 'save', 'saves'],
    'schools': ['school'],
    'ships': ['ship'],
    'shirts': ['shirt'],
    'situations': ['situation'],
    'slept': ['sleeping', 'sleep', 'sleeps'],
    'soldiers': ['soldier'],
    'specialized': ['specializing', 'specialize', 'specializes'],
    'sports': ['sport'],
    'students': ['student'],    
    'tailors': ['tailor'],
    'teachers': ['teacher'],
    'took': ['taking', 'take', 'takes'],
    'verified': ['verifying', 'verify', 'verifies'],
    'visited': ['visiting', 'visit', 'visits'],    
    'warriors': ['warrior'],
    'watered': ['watering', 'water', 'waters'],
    'worked': ['working', 'work', 'works'],
    'wrote': ['writing', 'write', 'writes']
    }

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
        elif len(query.split()) > 1:            
            term_word = 'phrase'
        
        if i == 0:
            color, color_text = GREEN, 'green'
        else:
            color, color_text = PURPLE, 'purple'

        constrained_query = '%s (in the sense of "%s")' % (query, constraint) if constraint else query
        line1 = 'Words related to the %s <b><font color="%s">%s</font></b> appear in <b><font color="%s">%s</font></b> in the summary.' % (term_word, color, constrained_query, color, color_text)        

        line2 = []
        exact_matches = [word for word in exact_match_list if word.lower() in query.lower()]
        num_matches = len(exact_matches)
        if num_matches > 0:
            match_word = 'word'
            if num_matches > 1:
                match_word = match_word + 's'
            line2.append('We found a match for the %s <b><font color="%s">%s</font></b>. It is highlighted in yellow.' % (match_word, color, ' '.join(exact_matches)))

        not_found = [word for word in not_found_list if word.lower() in query.lower()]
        num_missing = len(not_found)
        some_missing = num_missing > 0
        if some_missing:            
            missing_word = 'word'
            if num_missing > 1:
                missing_word = missing_word + 's'
            line2.append('We did not find a match for some words.')
            
        line2 = ' '.join(line2)

        line3 = ''
        if not some_missing: # all exact matches found
            if conceptual:                
                line3 = 'Check the summary to make sure it mentions the topic of <b><font color="%s">%s</font></b>.' % (color, constrained_query)
                if constraint:
                    line3 = line3 + ' Make sure to look for the specific sense of <b><font color="%s">%s</font></b> given by <b><font color="%s">"%s"</font></b>.' % (color, query, color, constraint)
            elif example:
                line3 = 'Check the summary to make sure it mentions an example of <b><font color="%s">%s</font></b>.' % (color, constrained_query)
                if constraint:
                    line3 = line3 + ' Make sure to look for the specific sense of <b><font color="%s">%s</font></b> given by <b><font color="%s">"%s"</font></b>.' % (color, query, color, constraint)
            elif constraint:
                line3 = 'Check the summary to make sure the specific sense of <b><font color="%s">%s</font></b> given by <b><font color="%s">"%s"</font></b> appears.' % (color, query, color, constraint)

        else: # some query terms not found
            if conceptual:
                line3 = 'Read all sections of the summary and look for words or phrases that are related to the topic of <b><font color="%s">%s</font></b>.' % (color, constrained_query)
                
            elif example:
                line3 = 'Read all sections of the summary and look for words or phrases that are examples of <b><font color="%s">%s</font></b>.' % (color, constrained_query)

            else:
                line3 = 'Read all sections of the summary and look for words or phrases that mean the same thing as the %s <b><font color="%s">%s</font></b>.' % (term_word, color, constrained_query)

            if morphological:
                if morphological in morph_alts:                
                    line3 = line3[:-1] + ' (NOT %s).' % ', '.join(morph_alts[morphological])
                else:
                    line3 = line3[:-1] + ' AND has exactly the same number (singluar vs. plural, for nouns) or tense (for verbs).'

            if constraint:
                line3 = line3 + ' Make sure to look for <b><font color="%s">%s</font></b> in the sense of "<b><font color="%s">"%s"</font></b>."' % (color, query, color, constraint)
            
        all_outputs.append('\n'.join([line1, line2, line3]))
    return all_outputs
