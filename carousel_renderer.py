from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import time


# --------------------------------------------------
# FONT LOADER (tries bold / regular safely)
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
# LOAD IMAGE FROM URL OR FALLBACK
# --------------------------------------------------
def load_background(image_url, width=1080, height=1080):
    try:
        r = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = width / height
        
        if img_ratio > target_ratio:
            # image is wider → crop sides
            new_height = height
            new_width = int(height * img_ratio)
        else:
            # image is taller → crop top/bottom
            new_width = width
            new_height = int(width / img_ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)

        
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        
        return img.crop((left, top, left + width, top + height))

    except:
        return Image.new("RGB", (width, height), (18, 18, 18))


# --------------------------------------------------
# SOFT GRADIENT (OPTIONAL, VERY SUBTLE)
# --------------------------------------------------
def apply_smart_gradient(img):
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # gradient starts from middle-lower area
    start_y = int(height * 0.55)

    for y in range(start_y, height):
        alpha = int(220 * ((y - start_y) / (height - start_y)) ** 1.1)
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# --------------------------------------------------
# WATERMARK AT BOTTOM CENTER
# --------------------------------------------------
def draw_branding(draw, width, height):
    text = "NEWS4YOU2026"
    font = get_font(26, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]

    draw.text(
        ((width - tw) / 2, height - 60),
        text,
        fill=(180, 180, 180),
        font=font
    )


# --------------------------------------------------
# TEXT MEASUREMENT HELPERS
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


# --------------------------------------------------
# DYNAMIC FONT SCALING FOR TITLE
# --------------------------------------------------
def fit_text_in_box(draw, text, start_size, min_size, bold,
                    max_width, max_height, line_spacing):
    size = start_size
    while size >= min_size:
        font = get_font(size, bold=bold)
        h, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
        if h <= max_height:
            return font, lines, h
        size -= 2

    font = get_font(min_size, bold=bold)
    h, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
    return font, lines, h


# --------------------------------------------------
# ROUNDED BACKGROUND CARD FOR TEXT
# --------------------------------------------------
def draw_text_background(base_img, x, y, w, h, radius=30, opacity=200):
    card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)

    d.rounded_rectangle(
        (0, 0, w, h),
        radius=radius,
        fill=(0, 0, 0, opacity)
    )

    base_img.paste(card, (x, y), card)

#splitting into slides

def split_text_into_slides(text, max_chars=350): # Increased to 350 for Slide 2 capacity
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

# --------------------------------------------------
# MAIN CAROUSEL GENERATOR
# --------------------------------------------------
def generate_carousel(article, topic):
    WIDTH, HEIGHT = 1080, 1080
    margin_x = 80
    max_width = WIDTH - (2 * margin_x)

    subtitle_font = get_font(32)
    body_font = get_font(44)
    meta_font = get_font(30, bold=True)

    slide_paths = []

    IMAGE_RATIO = 0.60
    IMAGE_HEIGHT = int(HEIGHT * IMAGE_RATIO)
    TEXT_HEIGHT = HEIGHT - IMAGE_HEIGHT


    # ================= SLIDE 1 =================
    # Create base canvas
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    
    # Load image ONLY for top 60%
    bg = load_background(article.get("image"), WIDTH, IMAGE_HEIGHT)
    
    # Paste image at top
    img.paste(bg, (0, 0))
    
    # Draw solid black background for bottom 40%
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        (0, IMAGE_HEIGHT, WIDTH, HEIGHT),
        fill=(0, 0, 0)
    )

    # HARD BLACK BAR BEHIND TOPIC (NO TRANSPARENCY)
    TOPIC_BAR_HEIGHT = 70
    
    draw.rectangle(
        (0, IMAGE_HEIGHT, WIDTH, IMAGE_HEIGHT + TOPIC_BAR_HEIGHT),
        fill=(0, 0, 0)
    )


    # ---------- SOLID IMAGE–TEXT DIVIDER ----------
    DIVIDER_HEIGHT = 8  # looks clean on Instagram
    
    draw.rectangle(
        (0, IMAGE_HEIGHT - DIVIDER_HEIGHT, WIDTH, IMAGE_HEIGHT),
        fill=(15, 15, 15)  # solid, clean break
    )



    draw_branding(draw, WIDTH, HEIGHT)

    # Text block positioning
    TEXT_BLOCK_Y = IMAGE_HEIGHT + 30

    TEXT_BLOCK_X = margin_x
    TEXT_BLOCK_WIDTH = WIDTH - (2 * TEXT_BLOCK_X)


    # Dynamic title
    TITLE_MAX_HEIGHT = 260
    title_font, title_lines, title_height = fit_text_in_box(
        draw,
        article.get("title", ""),
        start_size=68,
        min_size=42,
        bold=True,
        max_width=max_width,
        max_height=TITLE_MAX_HEIGHT,
        line_spacing=14
    )

    # Measure REAL subtitle height
    subtitle_exists = bool(article.get("desc"))
    subtitle_height = 0
    
    if subtitle_exists and title_font.size > 42:
        subtitle_height, _ = calculate_text_height(
            draw,
            article.get("desc")[:90] + "...",
            subtitle_font,
            max_width,
            10
        )

    # Calculate total card height
    CARD_PADDING = 30
    CARD_HEIGHT = (
        30 +               # topic
        title_height +
        subtitle_height +
        CARD_PADDING * 2
    )

    # Prevent card from overlapping watermark
    BOTTOM_BUFFER = 90
    MAX_Y = HEIGHT - CARD_HEIGHT - BOTTOM_BUFFER
    TEXT_BLOCK_Y = min(TEXT_BLOCK_Y, MAX_Y)


    # Draw background card
    # draw_text_background(
    #     img,
    #     TEXT_BLOCK_X,
    #     TEXT_BLOCK_Y,
    #     TEXT_BLOCK_WIDTH,
    #     CARD_HEIGHT
    # )

    # Draw text on top of card
    # ---------------- TOPIC IN SOLID BLACK BAR ----------------
    topic_y = IMAGE_HEIGHT + (TOPIC_BAR_HEIGHT - meta_font.size) // 2
    
    draw.text(
        (margin_x, topic_y),
        topic.upper(),
        fill="#FFD700",
        font=meta_font
    )

    # Start title BELOW the black bar
    y = IMAGE_HEIGHT + TOPIC_BAR_HEIGHT + 30


    # Title
    for line in title_lines:
        draw.text((margin_x, y), line, fill="white", font=title_font)
        y += title_font.getbbox(line)[3] + 14

    # Subtitle
    if subtitle_exists and title_font.size > 42:
        draw_wrapped_text(
            draw,
            article.get("desc")[:90] + "...",
            subtitle_font,
            margin_x,
            y + 12,
            max_width,
            "#E0E0E0",
            10
        )

    # Save slide 1
    p1 = f"slide_1_{int(time.time())}.png"
    img.save(p1, quality=95)
    slide_paths.append(p1)

# ---------- SLIDE 2 ----------
# Split the description into chunks that fit Slide 2
    raw_desc = article.get("desc", "").strip()
    if not raw_desc:
        return slide_paths
    description_chunks = split_text_into_slides(raw_desc, max_chars=400)


    for i, chunk in enumerate(description_chunks):
        img = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 18))
        draw = ImageDraw.Draw(img)
        draw_branding(draw, WIDTH, HEIGHT)
        # Header Info
        page_indicator = f" ({i+1}/{len(description_chunks)})" if len(description_chunks) > 1 else ""
        draw.text((margin_x, 80), topic.upper() + page_indicator, fill="#888888", font=meta_font)

        # Footer Source (only on last slide)
        if i == len(description_chunks) - 1:
            draw.text(
                (margin_x, HEIGHT - 130), 
                f"Source: {article.get('source', 'News')}",
                fill="#666666", font=meta_font
                )
        # Draw the chunk of text centered vertically
        total_h, _ = calculate_text_height(draw, chunk, body_font, max_width, 22)
        start_y = (HEIGHT - total_h) // 2
        
        draw_wrapped_text(draw, chunk, body_font, margin_x, start_y, max_width, "white", 22)

        # Accent bar
        bar_h = 500
        draw.rectangle((margin_x - 35, (HEIGHT - bar_h) // 2, margin_x - 25, (HEIGHT + bar_h) // 2), fill="#3aa0ff")

        p_chunk = f"slide_desc_{i}_{int(time.time())}.png"
        img.save(p_chunk, quality=95)
        slide_paths.append(p_chunk)
        time.sleep(0.1) # Prevent same filename

    return slide_paths

  

    
