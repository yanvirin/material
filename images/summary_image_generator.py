from PIL import Image, ImageFont, ImageDraw, ImageColor

COLOR_BLACK = ImageColor.getrgb("black")
COLOR_WHITE = ImageColor.getrgb("white")
COLOR_YELLOW = ImageColor.getrgb("yellow")
COLOR_ORANGE = ImageColor.getrgb("orange")
COLOR_GREEN = ImageColor.getrgb("green")
COLOR_PURPLE = ImageColor.getrgb("purple")

def wrap_line(line, max_width, font, highlight_weights=None):
    line = list(line)
    if highlight_weights:
        highlight_weights = list(highlight_weights)
    
    space_size = font.getsize(" ")[0]
    finished_lines = []
    
    if highlight_weights:
        finished_weights = []
    else:
        finished_weights = None

    while len(line):
        first_tok = line.pop(0)
        width = font.getsize(first_tok)[0]
        current_buffer = [first_tok]
        if highlight_weights:
            weight_buffer = [highlight_weights.pop(0)]
        else:
            weight_buffer = None
        
        while len(line):
            new_width = width + space_size + font.getsize(line[0])[0]
            if new_width < max_width:
                width = new_width
                current_buffer.append(line.pop(0))
                if weight_buffer:
                    weight_buffer.append(highlight_weights.pop(0))
            else:
                break

        finished_lines.append(current_buffer)
        if weight_buffer:
            finished_weights.append(weight_buffer)

    return finished_lines, finished_weights


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


def draw_wrapped_lines(wrapped_lines, draw, font, y_pos, wrapped_weights=None,
                       font_color=COLOR_WHITE,
                       highlight_color=COLOR_ORANGE):

    space_size = font.getsize(" ")[0]
    font_height = font.getsize("A")[1] + 5

    draw.text((10, y_pos), "\u2022", font=font, fill=font_color)
    for l, line in enumerate(wrapped_lines):
        x_pos = 25
        for t, token in enumerate(line):
            draw.text((x_pos, y_pos), token, font=font, fill=font_color)
            if wrapped_weights and wrapped_weights[l][t] > 0.:
                alpha = int(255 * wrapped_weights[l][t])
                draw.text((x_pos, y_pos), token, font=font, 
                          fill=(*highlight_color, alpha))

            x_pos += font.getsize(token)[0] + space_size
        y_pos += font_height 

     
def generate_image(path, summary_lines, topics=None, highlight_weights=None, 
                   width=600, height=600, 
                   font_color=COLOR_WHITE, background_color=COLOR_BLACK,
                   query_highlight_color=COLOR_GREEN, 
                   query_color=COLOR_PURPLE,
                   highlight_color=COLOR_GREEN):

    image = Image.new('RGBA', (width, height), background_color)

    font_size = 17
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 17)
    font_height = font.getsize("A")[1] + 5
    draw = ImageDraw.Draw(image)

    y_pos = 10

    if topics:
        for topic in topics:
            draw_topic(
                topic, draw, font, y_pos,
                query_color=query_color,
                query_highlight_color=query_highlight_color)
            y_pos += font_height
        y_pos += 20
        draw.line([(0, y_pos), (width, y_pos)], fill=font_color)
        y_pos += 20

    for b, bullet_item in enumerate(summary_lines):
        if highlight_weights:
            bullet_highlight_weights = highlight_weights[b]
        else:
            bullet_highlight_weights = None
        
        wrapped_lines, wrapped_weights = wrap_line(
            bullet_item, width - 25, font, 
            highlight_weights=bullet_highlight_weights)

        draw_wrapped_lines(wrapped_lines, draw, font, y_pos, 
                           wrapped_weights=wrapped_weights,
                           highlight_color=highlight_color)
        y_pos += (len(wrapped_lines) * font_height + 10)



    image.save(path, "PNG")
