import summary_image_generator

topics = [{"query": ["garlic"], "query_highlight": [True],
           "topic_words": ["onions", "garlic", "goods", "rice"]},
          {"query": ["pepper"], "query_highlight": [False],
           "topic_words": ["onions", "garlic", "goods"]}]

lines = [
    "There is a need, a store of rice is more than NFA rice than expensive rice , aside from having the government of monitoring team against illegal selling rice with NFA rice .".split(),
    "These soldiers even said, the smuggling of the country continues but President Aquino will not be able to identify the lack of rice supply , garlic , onions and other products when many are importing the international ports in the care of the Bureau of Customs and is near to rot.".split(),
    "It might be possible that NFA rice will".split()]

wmap = {"garlic": 1.0, "onions": .76, "goods": 45, "rice": .65}


highlight_weights = [[wmap.get(tok, 0.) for tok in line] for line in lines]


path = "../image_tmp/file2.png"
summary_image_generator.generate_image(path, lines, topics, highlight_weights)
#summary_image_generator.generate_image(lines, topics)
