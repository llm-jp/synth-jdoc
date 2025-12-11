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


def to_fullwidth(number):
    return str(number).translate(str.maketrans("0123456789", "０１２３４５６７８９"))


def apply_vertical_formatting(text):
    jp_chars = r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff01-\uff0f\uff1a-\uffef]'
    pattern = f'(?<={jp_chars})\d{{1,3}}(?={jp_chars})'
    def replace_tcy(match):
        return f'<span class="tcy">{unicodedata.normalize("NFKC", match.group(0))}</span>'
    return re.sub(pattern, replace_tcy, text)


def preprocess_elements(elements, is_vertical=False):
    processed_blocks = []
    current_multicolumn_block = []

    for element in elements:
        el = element.copy()
        
        if is_vertical and el.get('type') == 'text':
            el['content'] = apply_vertical_formatting(el['content'])

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

    font_families = ["'Noto Serif JP', serif", "'Noto Sans JP', sans-serif"]
    bg_colors = ["#ffffff", "#fafafa", "#fdfbf7", "#f0f0f0", "#fffff0"]
    figure_bg_colors = ["#eef", "#efe", "#fee", "#f0f8ff", "#faf0e6"]
    text_colors = ["#000000", "#1a1a1a", "#333333"]

    bg_color = random.choice(bg_colors)
    figure_bg_color = random.choice(figure_bg_colors) if random.random() > 0.5 else bg_color

    config = {
        "is_vertical": is_vertical,
        "writing_mode": "vertical-rl" if is_vertical else "horizontal-tb",
        "column_count": column_count,
        "padding": random.randint(30, 60),
        "font_family": random.choice(font_families),
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
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&family=Noto+Sans+JP:wght@400;700&display=swap');

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
        font-family: "'Noto Sans JP', sans-serif";
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
    <h1>{{ data.title }}</h1>
    
    {% for block in processed_blocks %}
        {% if block.type == 'multicolumn' %}
            <div class="content-body">
                {% for element in block.elements %}
                    {% if element.type == 'text' %}
                        <p>{{ element.content | safe }}</p>
                    
                    {% elif element.type == 'image' %}
                        {% if element.span_all and not style.is_vertical %}
                             <figure class="span-all">
                                <img src="{{ element.src }}" class="content-image" style="max-height:{{ style.max_img_limit }}px; object-fit:contain;">
                                {% if element.caption %}<figcaption>{{ element.caption }}</figcaption>{% endif %}
                            </figure>
                        {% else %}
                            <figure class="normal">
                                <img src="{{ element.src }}" class="content-image" style="{{ 'max-width' if style.is_vertical else 'max-height' }}: {{ style.max_img_limit }}px;">
                                {% if element.caption %}<figcaption>{{ element.caption }}</figcaption>{% endif %}
                            </figure>
                        {% endif %}
                    {% endif %}
                {% endfor %}
            </div>
        {% elif block.type == 'fullwidth' %}
            <figure class="span-all">
                <img src="{{ block.element.src }}" class="content-image" style="max-height:90%; max-width:{{ style.max_img_limit }}px;">
                {% if block.element.caption %}<figcaption>{{ block.element.caption }}</figcaption>{% endif %}
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
            content_data = copy.deepcopy(original_data)

            selected_formatter = random.choice(CAPTION_STYLES)
            image_idx = 0
            for el in content_data["elements"]:
                if el["type"] == "image":
                    image_idx += 1

                    image_path = os.path.join(image_dir, el["src"])
                    caption_path = os.path.join(caption_dir, os.path.splitext(el["src"])[0] + "_cap.txt")

                    if os.path.exists(image_path):
                        mime_type, _ = mimetypes.guess_type(image_path)
                        if mime_type is None:
                            mime_type = "image/png"
                        with open(image_path, "rb") as img_file:
                            b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                            el["src"] = f"data:{mime_type};base64,{b64_data}"
                    else:
                        raise ValueError(f"Image not found at {image_path}")

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

            # JSON保存
            label_data = {
                "id": base_filename,
                "elements": original_data['elements'],
                "style": style_config,
            }
            with open(os.path.join(output_dir, f"{base_filename}.json"), "w", encoding="utf-8") as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)

            # 画像生成
            await page.set_content(html_content, wait_until="networkidle")
            await page.screenshot(path=os.path.join(output_dir, f"{base_filename}.png"), full_page=True)


        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())