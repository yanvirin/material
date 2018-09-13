import pathlib
from PIL import Image, ImageFont, ImageDraw, ImageColor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


COLOR_BLACK = ImageColor.getrgb("black")
COLOR_WHITE = ImageColor.getrgb("white")
COLOR_YELLOW = ImageColor.getrgb("#EFEFE0")
COLOR_YELLOW = ImageColor.getrgb("yellow")
COLOR_ORANGE = ImageColor.getrgb("orange")
COLOR_GREEN = ImageColor.getrgb("#39FF14")
COLOR_RED = ImageColor.getrgb("red")
COLOR_PURPLE = ImageColor.getrgb("purple")
COLOR_ORCHID = ImageColor.getrgb("orchid")

def wrap_line(line, max_width, font, highlight_weights1=None, 
              highlight_weights2=None):
    line = list(line)
    if highlight_weights1:
        highlight_weights1 = list(highlight_weights1)
    if highlight_weights2:
        highlight_weights2 = list(highlight_weights2)
    
    space_size = font.getsize(" ")[0]
    finished_lines = []
    
    if highlight_weights1:
        finished_weights1 = []
    else:
        finished_weights1 = None

    if highlight_weights2:
        finished_weights2 = []
    else:
        finished_weights2 = None

    while len(line):
        first_tok = line.pop(0)
        width = font.getsize(first_tok)[0]
        current_buffer = [first_tok]
        if highlight_weights1:
            weight_buffer1 = [highlight_weights1.pop(0)]
        else:
            weight_buffer1 = None
        if highlight_weights2:
            weight_buffer2 = [highlight_weights2.pop(0)]
        else:
            weight_buffer2 = None
        
        while len(line):
            new_width = width + space_size + font.getsize(line[0])[0]
            if new_width < max_width:
                width = new_width
                current_buffer.append(line.pop(0))
                if weight_buffer1:
                    weight_buffer1.append(highlight_weights1.pop(0))
                if weight_buffer2:
                    weight_buffer2.append(highlight_weights2.pop(0))
            else:
                break

        finished_lines.append(current_buffer)
        if weight_buffer1:
            finished_weights1.append(weight_buffer1)
        if weight_buffer2:
            finished_weights2.append(weight_buffer2)

    return finished_lines, finished_weights1, finished_weights2


def draw_topic(topic, draw, font, y_pos, topic_color=COLOR_WHITE, 
               query_color=COLOR_YELLOW, 
               query_highlight_color=COLOR_ORANGE):

    space_size = font.getsize(" ")[0]
    x_pos = 10

    for q, query in enumerate(topic["query"]):
        if "query_highlight" in topic and topic["query_highlight"][q]:
            color = query_highlight_color
        else:
            color = query_color
        if q > 0:
            x_pos += space_size
        draw.text((x_pos, y_pos), query, font=font, fill=color)
        x_pos += font.getsize(query)[0]
    draw.text((x_pos, y_pos), ": " + " ".join(topic["topic_words"]), 
              font=font, fill=topic_color)


def draw_wrapped_lines(wrapped_lines, draw, font, bold_font, y_pos, 
                       wrapped_weights1=None,
                       wrapped_weights2=None,
                       font_color=COLOR_WHITE,
                       highlight_color1=COLOR_GREEN,
                       highlight_color2=COLOR_ORCHID,
                       exact_match_highlight=COLOR_RED):

    space_size = font.getsize(" ")[0]
    font_height = font.getsize("A")[1] + 5

    draw.text((10, y_pos), "\u2022", font=font, fill=font_color)
    for l, line in enumerate(wrapped_lines):
        x_pos = 25
        for t, token in enumerate(line):
            w, h = font.getsize(token)
            if wrapped_weights1 and wrapped_weights1[l][t] > 0.:
                if wrapped_weights1[l][t] == 1:
                    draw.rectangle(
                        (x_pos, y_pos, x_pos + w, y_pos + h), 
                        fill=exact_match_highlight)
                    draw.text((x_pos, y_pos), token, font=bold_font, 
                              #fill=highlight_color1)
                              fill=ImageColor.getrgb("green"))
                else:
                    draw.text((x_pos, y_pos), token, font=font, fill=font_color)
                    alpha = int(255 * wrapped_weights1[l][t])
                    draw.text((x_pos, y_pos), token, font=font, 
                              fill=(*highlight_color1, alpha))
            elif wrapped_weights2 and wrapped_weights2[l][t] > 0.:
                if wrapped_weights2[l][t] == 1:
                    draw.rectangle(
                        (x_pos, y_pos, x_pos + w, y_pos + h), 
                        fill=exact_match_highlight)
                    draw.text((x_pos, y_pos), token, font=bold_font, 
                            #  fill=highlight_color2)
                              fill=ImageColor.getrgb("purple"))
                else:
                    draw.text((x_pos, y_pos), token, font=font, fill=font_color)
                    alpha = int(255 * wrapped_weights2[l][t])
                    draw.text((x_pos, y_pos), token, font=font, 
                              fill=(*highlight_color2, alpha))
            else:
                draw.text((x_pos, y_pos), token, font=font, fill=font_color)

            x_pos += w + space_size
        y_pos += font_height 

     
def generate_image(path, summary_lines, topics=None, 
                   highlight_weights1=None,
                   highlight_weights2=None, 
                   width=1024, height=768, 
                   font_color=COLOR_WHITE, background_color=COLOR_BLACK,
                   query_highlight_color1=COLOR_GREEN, 
                   query_highlight_color2=COLOR_ORCHID, 
                   query_color=COLOR_WHITE,
                   highlight_color1=COLOR_GREEN,
                   highlight_color2=COLOR_ORCHID,
                   exact_match_highlight=COLOR_YELLOW,
                   missing_keywords=None):

    image = Image.new('RGBA', (width, height), background_color)

    font_size = 17
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 17)
    bold_font = ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 17)
    font_height = font.getsize("A")[1] + 5
    draw = ImageDraw.Draw(image)

    draw.text((10, 10), "RELATED WORDS", font=font, fill=font_color)
    y_pos = 10 + font_height + 10

    if topics:
        for t, topic in enumerate(topics):
            hlc = query_highlight_color1 if t == 0 else query_highlight_color2
            draw_topic(
                topic, draw, font, y_pos,
                query_color=query_color,
                query_highlight_color=hlc)
            y_pos += font_height
        y_pos += 20
        draw.line([(0, y_pos), (width, y_pos)], fill=font_color)
        y_pos += 20

    draw.text((10, y_pos), "SUMMARY", font=font, fill=font_color)
    y_pos += font_height + 10

    for b, bullet_item in enumerate(summary_lines):
        if highlight_weights1:
            bullet_highlight_weights1 = highlight_weights1[b]
        else:
            bullet_highlight_weights1 = None
        
        if highlight_weights2:
            bullet_highlight_weights2 = highlight_weights2[b]
        else:
            bullet_highlight_weights2 = None

        wrapped_lines, wrapped_weights1, wrapped_weights2 = wrap_line(
            bullet_item, width - 25, font, 
            highlight_weights1=bullet_highlight_weights1,
            highlight_weights2=bullet_highlight_weights2)

        draw_wrapped_lines(wrapped_lines, draw, font, bold_font, y_pos, 
                           wrapped_weights1=wrapped_weights1,
                           wrapped_weights2=wrapped_weights2,
                           highlight_color1=highlight_color1,
                           highlight_color2=highlight_color2,
                           exact_match_highlight=exact_match_highlight)
        y_pos += (len(wrapped_lines) * font_height + 10)
        
    draw.line([(0, y_pos + 10), (width, y_pos + 10)], fill=font_color)
    if missing_keywords:
        draw.text(
            (10, y_pos + 30),
            "WORDS NOT FOUND: " + " ".join(missing_keywords), 
            font=font, fill=font_color)

    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        image.save(fp, format="PNG")

def calculate_highlight_weights(query, summary_sentences, emb_dict,
                                stopwords, threshold_topk=3):

    normalized_summary_sents = [[w.lower() for w in sent]
                                for sent in summary_sentences]

    normalized_summary_words = list(set([w
                                         for sent in normalized_summary_sents
                                         for w in sent
                                         if w not in stopwords]))
    summary_word_embs = np.vstack(
        [emb_dict[w] for w in normalized_summary_words])

    query_words = [w.lower() for w in query]
    query_embs = np.vstack([emb_dict[w] for w in query_words])
    emb_sim = cosine_similarity(summary_word_embs, query_embs)

    query_thresholds = []
    for q, query in enumerate(query_words):
        thr = np.sort(emb_sim[:,q])[-threshold_topk]
        query_thresholds.append(thr)
    query_thresholds = np.array(query_thresholds)
    emb_sim[emb_sim < query_thresholds] = 0.
    word2emb_sim = {}
    for word, emb_sim_row in zip(normalized_summary_words, emb_sim):
        word2emb_sim[word] = np.max(emb_sim_row)

    highlight_weights = []
    for sent in normalized_summary_sents:
        highlight_weights.append([word2emb_sim.get(w, 0.0) for w in sent])

    # handle exact match
    qset = set(query_words)
    for s, sent in enumerate(normalized_summary_sents):
        for t, tok in enumerate(sent):
            if tok in qset:
                highlight_weights[s][t] = 1.0

    return highlight_weights
