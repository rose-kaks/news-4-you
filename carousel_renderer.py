from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import time


def get_font(size, bold=False):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()


def load_background(image_url, width=1080, height=1080):
    try:
        r = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img.resize((width, height))
    except:
        return Image.new("RGB", (width, height), (18, 18, 18))


def apply_smart_gradient(img):
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Start gradient at 75% height
    start_y = int(height * 0.75)

    for y in range(start_y, height):
        alpha = int(240 * ((y - start_y) / (height - start_y)) ** 1.3)
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")



def draw_branding(draw, width, height):
    text = "NEWS4YOU2026"
    font = get_font(26, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]

    # TOP CENTER instead of bottom
    draw.text(
        ((width - tw) / 2, 30),
        text,
        fill=(180, 180, 180),
        font=font
    )


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

def split_text_into_slides(text, max_chars=220):
    words = text.split()
    slides = []
    current = ""

    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current += " " + w
        else:
            slides.append(current.strip())
            current = w

    if current:
        slides.append(current.strip())

    return slides

def draw_wrapped_text(
    draw, text, font, x, y, max_width,
    fill="white", line_spacing=8,
    center_vertically=False, canvas_height=1080
):
    total_h, lines = calculate_text_height(draw, text, font, max_width, line_spacing)

    if center_vertically:
        y = (canvas_height - total_h) / 2

    cy = y
    for l in lines:
        draw.text((x, cy), l, fill=fill, font=font)
        cy += font.getbbox(l)[3] + line_spacing

    return cy

def fit_text_in_box(
    draw,
    text,
    font_size_start,
    font_size_min,
    bold,
    max_width,
    max_height,
    line_spacing
):
    font_size = font_size_start

    while font_size >= font_size_min:
        font = get_font(font_size, bold=bold)
        total_h, lines = calculate_text_height(
            draw, text, font, max_width, line_spacing
        )

        if total_h <= max_height:
            return font, lines, total_h

        font_size -= 2  # smooth scaling

    # fallback (smallest readable)
    font = get_font(font_size_min, bold=bold)
    total_h, lines = calculate_text_height(
        draw, text, font, max_width, line_spacing
    )
    return font, lines, total_h



def generate_carousel(article, topic):
    WIDTH, HEIGHT = 1080, 1080
    margin_x = 80
    max_width = WIDTH - (2 * margin_x)

    title_font = get_font(68, bold=True)
    subtitle_font = get_font(32, bold=False)
    body_font = get_font(44, bold=False)
    meta_font = get_font(30, bold=True)

    slide_paths = []

    # ---------- SLIDE 1 ----------

       
    bg = load_background(article.get("image"))
    img = apply_smart_gradient(bg)
    draw = ImageDraw.Draw(img)

    draw_branding(draw, WIDTH, HEIGHT)

    # Layout constants
    TEXT_START_Y = int(HEIGHT * 0.75) + 40
    TEXT_MARGIN_X = margin_x
    TEXT_MAX_WIDTH = max_width

    # Topic
    draw.text(
        (TEXT_MARGIN_X, TEXT_START_Y),
        topic.upper(),
        fill="#FFD700",
        font=meta_font
    )

    y = TEXT_START_Y + 55

    # ---------- TITLE (Dynamic Scaling) ----------
    TITLE_BOX_HEIGHT = int(HEIGHT * 0.25) - 140

    title_font, title_lines, title_height = fit_text_in_box(
        draw=draw,
        text=article.get("title", ""),
        font_size_start=68,
        font_size_min=42,
        bold=True,
        max_width=TEXT_MAX_WIDTH,
        max_height=TITLE_BOX_HEIGHT,
        line_spacing=14
    )

    cy = y
    for line in title_lines:
        draw.text((TEXT_MARGIN_X, cy), line, fill="white", font=title_font)
        cy += title_font.getbbox(line)[3] + 14

    y = cy

    # ---------- Subtitle ----------
    if y < HEIGHT - 120:
        subtitle = (article.get("desc", "")[:90] + "...") if article.get("desc") else ""
        draw_wrapped_text(
            draw,
            subtitle,
            subtitle_font,
            TEXT_MARGIN_X,
            y + 18,
            TEXT_MAX_WIDTH,
            "#E0E0E0",
            10
        )

    p1 = f"slide_1_{int(time.time())}.png"
    img.save(p1, quality=95)
    slide_paths.append(p1)


    # ---------- SLIDE 2 ----------
    img = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    line_spacing = 22
    description_text = (article.get("desc", "")[:200] + "...") if article.get("desc") else ""

    draw_branding(draw, WIDTH, HEIGHT)
    draw.text((margin_x, 80), topic.upper(), fill="#888888", font=meta_font)
    draw.text(
        (margin_x, HEIGHT - 130),
        f"Source: {article.get('source', 'News')}",
        fill="#666666",
        font=meta_font
    )

    draw_wrapped_text(
        draw,
        description_text,
        body_font,
        margin_x,
        0,
        max_width,
        "white",
        22,
        center_vertically=True,
        canvas_height=HEIGHT
    )

    bar_h = 500
    draw.rectangle(
        (margin_x - 35, (HEIGHT - bar_h) / 2,
         margin_x - 25, (HEIGHT + bar_h) / 2),
        fill="#3aa0ff"
    )

    p2 = f"slide_2_{int(time.time())}.png"
    img.save(p2, quality=95)
    slide_paths.append(p2)

    return slide_paths
