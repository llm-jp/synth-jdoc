import asyncio
import re
import random
import json
import os
import unicodedata
import copy
import base64
import mimetypes

from tqdm import tqdm
from playwright.async_api import async_playwright
from jinja2 import Template
from PIL import Image
from kanjize import number2kanji


CAPTION_STYLES = [
    # h=半角, f=全角, k=漢数字, t=キャプション, v=縦書きかどうか
    # 半角数字
    lambda h, f, k, t, v: f"図{h}: {t}",
    lambda h, f, k, t, v: f"図{h} : {t}",
    lambda h, f, k, t, v: f"図{h}.{t}",
    lambda h, f, k, t, v: f"図{h}. {t}",
    lambda h, f, k, t, v: f"Figure {h}: {t}",
    lambda h, f, k, t, v: f"Fig. {h}: {t}",
    lambda h, f, k, t, v: f"図{h} {t}",

    # 全角数字
    lambda h, f, k, t, v: f"図{f}: {t}",
    lambda h, f, k, t, v: f"図{f} : {t}",
    lambda h, f, k, t, v: f"図{f}：{t}",
    lambda h, f, k, t, v: f"図{f}{':' if v else '.'}{t}",
    lambda h, f, k, t, v: f"図{f}{':' if v else '.'} {t}",
    lambda h, f, k, t, v: f"図{f} {t}",
    lambda h, f, k, t, v: f"図{f}　{t}",
    
    # 漢数字
    lambda h, f, k, t, v: f"図{k}: {t}",
    lambda h, f, k, t, v: f"図{k} : {t}",
    lambda h, f, k, t, v: f"図{k}：{t}",
    lambda h, f, k, t, v: f"図{k}{':' if v else '.'}{t}",
    lambda h, f, k, t, v: f"図{k}{':' if v else '.'} {t}",
    lambda h, f, k, t, v: f"図{k} {t}",
    lambda h, f, k, t, v: f"図{k}　{t}",

    # 矢印
    lambda h, f, k, t, v: f"{'▶' if v else '▲'} {t}",
]
FIG_PATTERN = r"^図\s*(?:\d+[：:\.\s]*|[：:\.\s]+)"
FONT_CANDIDATES = [
    # sans-serif
    {"name": "BIZ UDGothic", "family": "sans-serif"},
    {"name": "BIZ UDPGothic", "family": "sans-serif"},
    {"name": "Dela Gothic One", "family": "sans-serif"},
    {"name": "DotGothic16", "family": "sans-serif"},
    {"name": "IBM Plex Sans JP", "family": "sans-serif"},
    {"name": "Kiwi Maru", "family": "serif"},
    {"name": "Kosugi", "family": "sans-serif"},
    {"name": "Kosugi Maru", "family": "sans-serif"},
    {"name": "M PLUS 1", "family": "sans-serif"},
    {"name": "M PLUS 1p", "family": "sans-serif"},
    {"name": "M PLUS 2", "family": "sans-serif"},
    {"name": "M PLUS Rounded 1c", "family": "sans-serif"},
    {"name": "Mochiy Pop One", "family": "sans-serif"},
    {"name": "Mochiy Pop P One", "family": "sans-serif"},
    {"name": "Murecho", "family": "sans-serif"},
    {"name": "Noto Sans JP", "family": "sans-serif"},
    {"name": "Rampart One", "family": "sans-serif"},
    {"name": "RocknRoll One", "family": "sans-serif"},
    {"name": "Sawarabi Gothic", "family": "sans-serif"},
    {"name": "Shippori Antique", "family": "sans-serif"},
    {"name": "Shippori Antique B1", "family": "sans-serif"},
    {"name": "Stick", "family": "sans-serif"},
    {"name": "Yusei Magic", "family": "sans-serif"},
    {"name": "Zen Kaku Gothic Antique", "family": "sans-serif"},
    {"name": "Zen Kaku Gothic New", "family": "sans-serif"},
    {"name": "Zen Kurenaido", "family": "sans-serif"},
    {"name": "Zen Maru Gothic", "family": "sans-serif"},
    # serif
    {"name": "BIZ UDMincho", "family": "serif"},
    {"name": "BIZ UDPMincho", "family": "serif"},
    {"name": "Hina Mincho", "family": "serif"},
    {"name": "Kaisei Decol", "family": "serif"},
    {"name": "Kaisei HarunoUmi", "family": "serif"},
    {"name": "Kaisei Opti", "family": "serif"},
    {"name": "Kaisei Tokumin", "family": "serif"},
    {"name": "New Tegomin", "family": "serif"},
    {"name": "Noto Serif JP", "family": "serif"},
    {"name": "Shippori Mincho", "family": "serif"},
    {"name": "Shippori Mincho B1", "family": "serif"},
    {"name": "Yuji Boku", "family": "serif"},
    {"name": "Yuji Mai", "family": "serif"},
    {"name": "Yuji Syuku", "family": "serif"},
    {"name": "Zen Antique", "family": "serif"},
    {"name": "Zen Antique Soft", "family": "serif"},
    {"name": "Zen Old Mincho", "family": "serif"},
    # cursive
    {"name": "Hachi Maru Pop", "family": "cursive"},
    {"name": "Klee One", "family": "cursive"},
    # monospace
    {"name": "M PLUS 1 Code", "family": "monospace"},
    # system-ui
    {"name": "Potta One", "family": "system-ui"},
    {"name": "Reggae One", "family": "system-ui"},
]


def to_fullwidth(number):
    return str(number).translate(str.maketrans("0123456789", "０１２３４５６７８９"))


def remove_outer_brackets(text: str, start_char: str = "「", end_char: str = "」"):
    """
    文字列全体が「」で囲まれている場合のみ、その両端の「」を除去する。
    `「A」と「B」` のように、途中で括弧が閉じる場合は除去しない。
    """
    text = text.strip()
    if not (text.startswith(start_char) and text.endswith(end_char)):
        return text

    depth = 0
    for i, char in enumerate(text[:-1]):
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
        
        # 最初の「 が、文字列の途中で閉じてしまった場合
        if depth == 0 and i > 0:
            return text

    # 最後まで depth が 0 にならずに到達できた場合のみ除去
    return text[1:-1]


def apply_vertical_formatting(text):
    jp_chars = r'[ \u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff01-\uff0f\uff1a-\uffef]'
    pattern = f'(?:^|(?<={jp_chars}))\d{{1,3}}(?={jp_chars})'
    def replace_tcy(match):
        return f'<span class="tcy">{unicodedata.normalize("NFKC", match.group(0))}</span>'
    return re.sub(pattern, replace_tcy, text)


def preprocess_elements(elements, is_vertical=False):
    processed_blocks = []
    current_multicolumn_block = []

    for i, element in enumerate(elements):
        el = element.copy()
        el['id'] = i
        
        if is_vertical and el.get('type') == 'text':
            el['content'] = apply_vertical_formatting(el['content'])
        
        if is_vertical and el.get('type') == 'image' and el.get('caption'):
            el['caption'] = apply_vertical_formatting(el['caption'])

        is_fullwidth_image = (el.get('type') == 'image' and el.get('span_all') is True)

        if is_vertical and is_fullwidth_image:
            if current_multicolumn_block:
                processed_blocks.append({'type': 'multicolumn', 'elements': current_multicolumn_block})
                current_multicolumn_block = []
            processed_blocks.append({'type': 'fullwidth', 'element': el})
        else:
            current_multicolumn_block.append(el)

    if current_multicolumn_block:
        processed_blocks.append({'type': 'multicolumn', 'elements': current_multicolumn_block})

    return processed_blocks


def generate_random_style_config(**kwargs):
    is_vertical = random.choice([True, False])
    column_count = random.randint(1, 3)

    bg_colors = ["#ffffff", "#fafafa", "#fdfbf7", "#f0f0f0", "#fffff0"]
    figure_bg_colors = ["#eef", "#efe", "#fee", "#f0f8ff", "#faf0e6"]
    text_colors = ["#000000", "#1a1a1a", "#333333"]

    selected_font = random.choice(FONT_CANDIDATES)
    font_name = selected_font["name"]
    font_generic_family = selected_font["family"]

    font_family_css = f"'{font_name}', {font_generic_family}"
    font_param = font_name.replace(" ", "+")
    font_url = f"https://fonts.googleapis.com/css2?family={font_param}:wght@400&display=swap"

    bg_color = random.choice(bg_colors)
    figure_bg_color = random.choice(figure_bg_colors) if random.random() > 0.5 else bg_color

    show_title = random.choice([True, False])

    config = {
        "is_vertical": is_vertical,
        "writing_mode": "vertical-rl" if is_vertical else "horizontal-tb",
        "column_count": column_count,
        "padding": random.randint(30, 60),
        "font_family": font_family_css,
        "font_url": font_url,
        "base_font_size": random.randint(16, 22),
        "line_height": round(random.uniform(1.6, 2.0), 1),
        "bg_color": bg_color,
        "figure_bg_color": figure_bg_color,
        "text_color": random.choice(text_colors),
        "column_gap": random.randint(30, 60),
        "rule_style": "1px solid #ddd" if column_count > 1 else "none",
        "h1_border_style": random.choice(["solid", "double", "dashed"]),
        "h1_border_width": random.randint(2, 6),
        "image_alignment": random.choice(["left", "center"]),
        "show_title": show_title,
    }

    config.update(kwargs)
    config["writing_mode"] = "vertical-rl" if config["is_vertical"] else "horizontal-tb"

    if config["is_vertical"]:
        view_w, view_h = 100, 800 + 200 * (config["column_count"] - 1)
        total_len = view_h
        padding = config["padding"] * 2
        gap = config["column_gap"] * (config["column_count"] - 1) if config["column_count"] > 1 else 0
        col_dim = (total_len - padding - gap) / config["column_count"]
        
        config["max_img_limit"] = random.randint(
            min(int(view_h * (1/3)), int(col_dim * (2/3 + 1/3 * (config["column_count"] - 1)))),
            max(int(view_h * (1/3)), int(col_dim * (2/3 + 1/3 * (config["column_count"] - 1)))),
        )
    else:
        view_w, view_h = 800 + 200 * (config["column_count"] - 1), 100
        total_len = view_w
        padding = config["padding"] * 2
        gap = config["column_gap"] * (config["column_count"] - 1) if config["column_count"] > 1 else 0
        col_dim = (total_len - padding - gap) / config["column_count"]

        config["max_img_limit"] = random.randint(
            min(int(view_w * (1/3)), int(col_dim * (2/3 + 1/3 * (config["column_count"] - 1)))),
            max(int(view_w * (1/3)), int(col_dim * (2/3 + 1/3 * (config["column_count"] - 1)))),
        )

    config["viewport_width"] = view_w
    config["viewport_height"] = view_h

    return config


html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{{ style.font_url }}" rel="stylesheet">
<style>
    html { height: 100%; margin: 0; }

    body {
        margin: 0;
        padding: {{ style.padding }}px;
        background-color: {{ style.bg_color }};
        color: {{ style.text_color }};
        font-family: {{ style.font_family }};
        writing-mode: {{ style.writing_mode }};
        text-orientation: mixed;
        box-sizing: border-box;

        {% if style.is_vertical %}
            height: 100%;
            width: auto;
        {% else %}
            height: auto;
            min-height: 100%;
            width: 100%;
        {% endif %}
    }

    h1 {
        font-size: {{ style.base_font_size * 1.6 }}px;
        font-weight: 700;
        border-left: {{ style.h1_border_width }}px {{ style.h1_border_style }} {{ style.text_color }};
        padding-left: 15px;
        margin-left: 10px;
        margin-bottom: 30px;
        flex-shrink: 0;
    }

    .content-body {
        column-count: {{ style.column_count }};
        column-gap: {{ style.column_gap }}px;
        column-rule: {{ style.rule_style }};
        font-size: {{ style.base_font_size }}px;
        line-height: {{ style.line_height }};
        text-align: justify;
        column-fill: balance;
        
        {% if style.is_vertical %}
            height: calc(100vh - {{ style.padding * 2 }}px);
            width: auto;
        {% else %}
            width: 100%;
            display: block;
        {% endif %}
    }

    p { margin-top: 0; margin-bottom: 1em; text-indent: 1em; }

    .tcy {
        text-combine-upright: all;
        font-family: {{ style.font_family }};
        margin: 2px 0;
    }

    figure {
        display: block;
        background-color: rgba(0,0,0,0.03);
        padding: 5px;
        text-align: {{ style.image_alignment }};
        margin: 0;
        break-inside: avoid;
        box-sizing: border-box;
    }
    
    .content-image { 
        max-width: 100%; 
        max-height: 100%; 
        width: auto; 
        height: auto; 
        /* border: 1px solid #ccc; */
        object-fit: contain;
        display: block;
        /* margin: 0 auto; */
    }

    figure.normal .content-image {
        {% if style.image_alignment == 'center' %}
            margin-inline: auto; /* 横書きなら左右、縦書きなら上下を自動で中央揃え */
        {% else %}
            margin-inline: 0;    /* 左寄せ */
        {% endif %}
    }

    figure.span-all .content-image {
        margin-inline: auto;
    }
    
    figcaption { font-size: 0.8em; font-weight: bold; margin-top: 5px; }

    figure.normal {
        background-color: {{ style.figure_bg_color }};
        {% if style.is_vertical %}
            max-height: 100%;
            max-width: 100%;
            width: auto;
            height: auto;
            margin-left: auto;
            margin-right: auto;
        {% else %}
            width: 100%;
            height: auto;
        {% endif %}
    }
    
    figure.span-all {
        display: block; 
        background-color: {{ style.figure_bg_color }};
        box-sizing: border-box;
        text-align: center;
        
        {% if style.is_vertical %}
            height: calc(100vh - {{ style.padding * 2 }}px);
            width: auto;
            margin: 0 {{ style.column_gap }}px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        {% else %}
            column-span: all;
            width: 100%;
            height: auto;
            margin: 30px 0;
            padding: 10px;
            text-align: center;
        {% endif %}
    }
</style>
</head>
<body>
    {% if style.show_title %}
        <h1 data-id="title">{{ data.title }}</h1>
    {% endif %}
    {% for block in processed_blocks %}
        {% if block.type == 'multicolumn' %}
            <div class="content-body">
                {% for element in block.elements %}
                    {% if element.type == 'text' %}
                        <p data-id="{{ element.id }}">{{ element.content | safe }}</p>
                    
                    {% elif element.type == 'image' %}
                        {% if element.span_all and not style.is_vertical %}
                             <figure class="span-all" data-id="{{ element.id }}">
                                <img src="{{ element.src }}" class="content-image" style="max-height:{{ style.max_img_limit }}px; object-fit:contain;">
                                {% if element.caption %}<figcaption>{{ element.caption | safe }}</figcaption>{% endif %}
                            </figure>
                        {% else %}
                            <figure class="normal" data-id="{{ element.id }}">
                                <img src="{{ element.src }}" class="content-image" style="{{ 'max-width' if style.is_vertical else 'max-height' }}: {{ style.max_img_limit }}px;">
                                {% if element.caption %}<figcaption>{{ element.caption | safe }}</figcaption>{% endif %}
                            </figure>
                        {% endif %}
                    {% endif %}
                {% endfor %}
            </div>
        {% elif block.type == 'fullwidth' %}
            <figure class="span-all" data-id="{{ block.element.id }}">
                <img src="{{ block.element.src }}" class="content-image" style="max-height:90%; max-width:{{ style.max_img_limit }}px;">
                {% if block.element.caption %}<figcaption>{{ block.element.caption | safe }}</figcaption>{% endif %}
            </figure>
        {% endif %}
    {% endfor %}
</body>
</html>
"""


async def main():
    output_dir = "data/JSSODa/html_images"
    image_dir = "data/JSSODa/images"
    caption_dir = "data/JSSODa/captions"
    title_dir = "data/JSSODa/titles"

    os.makedirs(output_dir, exist_ok=True)

    data_list = []
    with open("data/JSSODa/jssoda_train_img_match.jsonl") as f:
        for line in f:
            data_list.append(json.loads(line))
    data_list = data_list[:50]

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        template = Template(html_template)

        for original_data in tqdm(data_list, desc="Generating images"):
            # load title
            with open(
                os.path.join(
                    title_dir,
                    f"{original_data['id'][:3]}/{original_data['id']}_title.txt",
                )
            ) as f:
                raw_title = f.read()
            original_data["title"] = remove_outer_brackets(raw_title)

            content_data = copy.deepcopy(original_data)

            caption_dict = {}
            selected_formatter = random.choice(CAPTION_STYLES)
            image_idx = 0
            for el in content_data["elements"]:
                if el["type"] == "image":
                    image_idx += 1

                    # load caption
                    caption_path = os.path.join(caption_dir, os.path.splitext(el["src"])[0] + "_cap.txt")
                    if os.path.exists(caption_path):
                        with open(caption_path) as f:
                            cap = f.read()
                    else:
                        raise ValueError(f"Caption not found at {caption_path}")

                    cap = re.sub(FIG_PATTERN, "", cap)
                    val_kanji = number2kanji(image_idx)
                    val_full = to_fullwidth(image_idx)
                    val_half = image_idx
                    cap = selected_formatter(val_half, val_full, val_kanji, cap, content_data["is_vertical"])
                    el["caption"] = cap
                    caption_dict[el["src"]] = cap

                    # load image
                    image_path = os.path.join(image_dir, el["src"])
                    if os.path.exists(image_path):
                        mime_type, _ = mimetypes.guess_type(image_path)
                        if mime_type is None:
                            mime_type = "image/png"
                        with open(image_path, "rb") as img_file:
                            b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                            el["src"] = f"data:{mime_type};base64,{b64_data}"
                    else:
                        raise ValueError(f"Image not found at {image_path}")

                    img = Image.open(image_path)
                    width, height = img.size

                    el["span_all"] = False
                    if content_data["is_vertical"] and (height > width):
                        if random.random() >= 0.5:
                            el["span_all"] = True
                    elif (not content_data["is_vertical"]) and (height < width):
                        if random.random() >= 0.5:
                            el["span_all"] = True

            # スタイルをランダム生成
            style_config = generate_random_style_config(
                is_vertical=content_data["is_vertical"],
                column_count=content_data["column_count"]
            )

            # ページの作成
            context = await browser.new_context(
                viewport={
                    'width': style_config["viewport_width"], 
                    'height': style_config["viewport_height"]
                }
            )
            page = await context.new_page()

            processed_blocks = preprocess_elements(
                content_data['elements'], 
                is_vertical=style_config['is_vertical']
            )

            html_content = template.render(
                data=content_data,
                processed_blocks=processed_blocks,
                style=style_config
            )

            base_filename = content_data["id"]
            
            # HTML保存
            with open(os.path.join(output_dir, f"{base_filename}.html"), "w", encoding="utf-8") as f:
                f.write(html_content)

            await page.set_content(html_content, wait_until="networkidle")

            bboxes_map = await page.evaluate('''() => {
                const results = {};
                const rootRect = document.documentElement.getBoundingClientRect();
                
                const isInside = (x, y, rect) => {
                    return (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom);
                };

                const createLineBoxes = (rects) => {
                    const list = [];
                    for (let i = 0; i < rects.length; i++) {
                        const r = rects[i];
                        if (r.width > 0 && r.height > 0) {
                            list.push({
                                rawRect: r,
                                x: r.x - rootRect.x,
                                y: r.y - rootRect.y,
                                width: r.width,
                                height: r.height,
                                text: ""
                            });
                        }
                    }
                    return list;
                };
                
                const createCharBoxes = (rootNode) => {
                    const list = [];
                    const treeWalker = document.createTreeWalker(rootNode, NodeFilter.SHOW_TEXT, null, false);
                    let textNode;
                    while (textNode = treeWalker.nextNode()) {
                        const str = textNode.nodeValue;
                        for (let i = 0; i < str.length; i++) {
                            const range = document.createRange();
                            range.setStart(textNode, i);
                            range.setEnd(textNode, i + 1);
                            const rect = range.getBoundingClientRect();
                            
                            if (rect.width > 0 && rect.height > 0) {
                                list.push({
                                    text: str[i],
                                    x: rect.x - rootRect.x,
                                    y: rect.y - rootRect.y,
                                    width: rect.width,
                                    height: rect.height
                                });
                            }
                        }
                    }
                    return list;
                };

                const mapTextToLineBoxes = (rootNode, boxes) => {
                    const treeWalker = document.createTreeWalker(rootNode, NodeFilter.SHOW_TEXT, null, false);
                    let textNode;
                    while (textNode = treeWalker.nextNode()) {
                        const str = textNode.nodeValue;
                        for (let i = 0; i < str.length; i++) {
                            const range = document.createRange();
                            range.setStart(textNode, i);
                            range.setEnd(textNode, i + 1);
                            const charRect = range.getBoundingClientRect();
                            const cx = charRect.left + charRect.width / 2;
                            const cy = charRect.top + charRect.height / 2;

                            for (let b = 0; b < boxes.length; b++) {
                                if (isInside(cx, cy, boxes[b].rawRect)) {
                                    boxes[b].text += str[i];
                                    break;
                                }
                            }
                        }
                    }
                };

                document.querySelectorAll('[data-id]').forEach(el => {
                    const id = el.getAttribute('data-id');
                    let lineBoxes = [];
                    let charBoxes = [];

                    if (el.tagName === 'FIGURE') {
                        // === 画像要素 ===
                        const img = el.querySelector('img');
                        if (img) {
                            lineBoxes.push(...createLineBoxes(img.getClientRects()));
                        }
                        
                        // === キャプション要素 ===
                        const figcaption = el.querySelector('figcaption');
                        if (figcaption) {
                            // 1. Line/Block Boxes
                            const range = document.createRange();
                            range.selectNodeContents(figcaption);
                            const capLineBoxes = createLineBoxes(range.getClientRects());
                            mapTextToLineBoxes(figcaption, capLineBoxes);

                            // 行統合処理
                            if (capLineBoxes.length > 0) {
                                let minX = Infinity, minY = Infinity;
                                let maxX = -Infinity, maxY = -Infinity;
                                let combinedText = "";

                                capLineBoxes.forEach(box => {
                                    if (box.x < minX) minX = box.x;
                                    if (box.y < minY) minY = box.y;
                                    if (box.x + box.width > maxX) maxX = box.x + box.width;
                                    if (box.y + box.height > maxY) maxY = box.y + box.height;
                                    combinedText += box.text;
                                });

                                lineBoxes.push({
                                    x: minX,
                                    y: minY,
                                    width: maxX - minX,
                                    height: maxY - minY,
                                    text: combinedText
                                });
                            }
                            
                            // 2. Char Boxes
                            charBoxes = createCharBoxes(figcaption);
                        }
                    } else {
                        // 1. Line/Block Boxes
                        lineBoxes = createLineBoxes(el.getClientRects());
                        mapTextToLineBoxes(el, lineBoxes);
                        
                        // 2. Char Boxes
                        charBoxes = createCharBoxes(el);
                    }

                    lineBoxes = lineBoxes.map(line => {
                        let minX = Infinity;
                        let minY = Infinity;
                        let maxX = -Infinity;
                        let maxY = -Infinity;
                        let hasChars = false;

                        // この行(line)に含まれる文字(char)を探して、その包含矩形を計算
                        for (let i = 0; i < charBoxes.length; i++) {
                            const c = charBoxes[i];
                            // 厳密な包含ではなく、交差(Intersection)判定で所属を確認
                            const intersects = (
                                c.x < line.x + line.width &&
                                c.x + c.width > line.x &&
                                c.y < line.y + line.height &&
                                c.y + c.height > line.y
                            );

                            if (intersects) {
                                hasChars = true;
                                if (c.x < minX) minX = c.x;
                                if (c.y < minY) minY = c.y;
                                if (c.x + c.width > maxX) maxX = c.x + c.width;
                                if (c.y + c.height > maxY) maxY = c.y + c.height;
                            }
                        }

                        // 対応する文字が見つかった場合のみ座標を更新
                        if (hasChars) {
                            return {
                                ...line, // 元のプロパティ(textなど)を維持
                                x: minX,
                                y: minY,
                                width: maxX - minX,
                                height: maxY - minY
                            };
                        }
                        
                        // 文字が見つからない（スペースのみの行など）場合は元のまま返す
                        return line;
                    });

                    results[id] = {
                        lines: lineBoxes.map(({rawRect, ...rest}) => rest),
                        chars: charBoxes
                    };
                });
                return results;
            }''')

            final_elements = []
            for i, el in enumerate(original_data['elements']):
                el_copy = el.copy()

                if str(i) in bboxes_map:
                    bbox_data = bboxes_map[str(i)]
                    el_copy['bboxes'] = bbox_data.get('lines', [])
                    el_copy['char_bboxes'] = bbox_data.get('chars', [])
                else:
                    el_copy['bboxes'] = []
                    el_copy['char_bboxes'] = []
                
                if el_copy['type'] == 'image':
                    el_copy['caption'] = caption_dict[el_copy['src']]

                final_elements.append(el_copy)

            # 画像生成
            await page.screenshot(path=os.path.join(output_dir, f"{base_filename}.png"), full_page=True)

            title_bboxes = []
            title_char_bboxes = []
            if "title" in bboxes_map:
                title_bboxes = bboxes_map["title"].get("lines", [])
                title_char_bboxes = bboxes_map["title"].get("chars", [])

            label_data = {
                "id": base_filename,
                "style": style_config,
                "title": content_data["title"],
                "title_bboxes": title_bboxes,
                "title_char_bboxes": title_char_bboxes,
                "elements": final_elements,
            }
            with open(os.path.join(output_dir, f"{base_filename}.json"), "w", encoding="utf-8") as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())