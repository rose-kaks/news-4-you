from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import time


# --------------------------------------------------
# FONT LOADER
# --------------------------------------------------
def get_font(size, bold=False):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------
def load_background(image_url, width=1080, height=1080):
    try:
        r = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            new_height = height
            new_width = int(height * img_ratio)
        else:
            new_width = width
            new_height = int(width / img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        left = (new_width - width) // 2
        top = (new_height - height) // 2
        return img.crop((left, top, left + width, top + height))
    except:
        return Image.new("RGB", (width, height), (18, 18, 18))


# --------------------------------------------------
# WATERMARK
# --------------------------------------------------
def draw_branding(draw, width, height):
    text = "NEWS4YOU2026"
    font = get_font(26, bold=True)
    tw = draw.textbbox((0, 0), text, font=font)[2]
    draw.text(((width - tw) / 2, height - 60), text, fill=(180, 180, 180), font=font)


# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------
def calculate_text_height(draw, text, font, max_width, line_spacing):
    words = text.split()
    lines, line = [], ""

    for w in words:
        test = f"{line} {w}".strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)

    height = sum(font.getbbox(l)[3] for l in lines) + (len(lines) - 1) * line_spacing
    return height, lines


def draw_wrapped_text(draw, text, font, x, y, max_width, fill, line_spacing):
    _, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
    cy = y
    for l in lines:
        draw.text((x, cy), l, fill=fill, font=font)
        cy += font.getbbox(l)[3] + line_spacing
    return cy


def fit_text_in_box(draw, text, start_size, min_size, bold, max_width, max_height, line_spacing):
    size = start_size
    while size >= min_size:
        font = get_font(size, bold)
        h, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
        if h <= max_height:
            return font, lines, h
        size -= 2
    font = get_font(min_size, bold)
    h, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
    return font, lines, h


def split_text_into_slides(text, max_chars=400):
    words, slides, current = text.split(), [], []
    length = 0
    for w in words:
        if length + len(w) + 1 <= max_chars:
            current.append(w)
            length += len(w) + 1
        else:
            slides.append(" ".join(current))
            current, length = [w], len(w)
    if current:
        slides.append(" ".join(current))
    return slides


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def generate_carousel(article, topic):
    WIDTH, HEIGHT = 1080, 1080
    margin_x = 80
    max_width = WIDTH - (2 * margin_x)

    WATERMARK_SPACE = 110

    subtitle_font = get_font(32)
    body_font = get_font(44)
    meta_font = get_font(30, bold=True)

    slide_paths = []

    IMAGE_HEIGHT = int(HEIGHT * 0.6)

    # ================= SLIDE 1 =================
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    bg = load_background(article.get("image"), WIDTH, IMAGE_HEIGHT)
    img.paste(bg, (0, 0))

    draw = ImageDraw.Draw(img)
    draw.rectangle((0, IMAGE_HEIGHT, WIDTH, HEIGHT), fill=(0, 0, 0))

    TOPIC_BAR_HEIGHT = 70
    draw.rectangle((0, IMAGE_HEIGHT, WIDTH, IMAGE_HEIGHT + TOPIC_BAR_HEIGHT), fill=(0, 0, 0))
    draw.rectangle((0, IMAGE_HEIGHT - 8, WIDTH, IMAGE_HEIGHT), fill=(15, 15, 15))

    draw_branding(draw, WIDTH, HEIGHT)

    MAX_TEXT_Y = HEIGHT - WATERMARK_SPACE

    title_font, title_lines, _ = fit_text_in_box(
        draw, article.get("title", ""),
        68, 42, True, max_width, 260, 14
    )

    subtitle_text = article.get("desc", "")[:90] + "..."
    subtitle_h, _ = calculate_text_height(draw, subtitle_text, subtitle_font, max_width, 10)

    draw.text(
        (margin_x, IMAGE_HEIGHT + (TOPIC_BAR_HEIGHT - meta_font.size) // 2),
        topic.upper(),
        fill="#FFD700",
        font=meta_font
    )

    y = IMAGE_HEIGHT + TOPIC_BAR_HEIGHT + 30
    for line in title_lines:
        lh = title_font.getbbox(line)[3]
        if y + lh > MAX_TEXT_Y:
            break
        draw.text((margin_x, y), line, fill="white", font=title_font)
        y += lh + 14

    if y + subtitle_h < MAX_TEXT_Y:
        draw_wrapped_text(draw, subtitle_text, subtitle_font, margin_x, y + 12, max_width, "#E0E0E0", 10)

    p1 = f"slide_1_{int(time.time())}.png"
    img.save(p1, quality=95, optimize=True)
    slide_paths.append(p1)

    # ================= SLIDE 2+ =================
    chunks = split_text_into_slides(article.get("desc", ""), 400)

    for i, chunk in enumerate(chunks):
        img = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 18))
        draw = ImageDraw.Draw(img)
        draw_branding(draw, WIDTH, HEIGHT)

        TOP_SAFE = 160
        BOTTOM_SAFE = 160

        label = f"{topic.upper()} ({i+1}/{len(chunks)})" if len(chunks) > 1 else topic.upper()
        draw.text((margin_x, 80), label, fill="#888888", font=meta_font)

        total_h, _ = calculate_text_height(draw, chunk, body_font, max_width, 22)
        usable_h = HEIGHT - TOP_SAFE - BOTTOM_SAFE
        centered_y = TOP_SAFE + (usable_h - total_h) // 2

        draw_wrapped_text(draw, chunk, body_font, margin_x, centered_y, max_width, "white", 22)

        draw.rectangle(
            (margin_x - 35, (HEIGHT - 500)//2, margin_x - 25, (HEIGHT + 500)//2),
            fill="#3aa0ff"
        )

        if i == len(chunks) - 1:
            draw.text((margin_x, HEIGHT - 130), f"Source: {article.get('source', 'News')}", fill="#666666", font=meta_font)
        else:
            draw.text((WIDTH - 150, HEIGHT - 150), "→", fill="#3aa0ff", font=get_font(50, True))

        p = f"slide_content_{i}_{int(time.time())}.png"
        img.save(p, quality=95, optimize=True)
        slide_paths.append(p)
        time.sleep(0.1)

        if len(slide_paths) >= 10:
            break

    return slide_paths
