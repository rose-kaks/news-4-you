from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import time


# --------------------------------------------------
# FONT LOADER
# --------------------------------------------------
def get_font(size, bold=False):
     """
    Attempts to load a TrueType font with the specified size and weight.
    Falls back to default font if no TrueType fonts are available.
    
    Args:
        size: Font size in points
        bold: Whether to use bold variant
    
    Returns:
        ImageFont object
    """
    # List of font paths to try, in order of preference
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf"
    ]
    # Try each font path until one works
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
     # If all else fails, use PIL's default font
    return ImageFont.load_default()


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------
def load_background(image_url, width=1080, height=1080):
    """
    Downloads an image from URL, resizes it to fill the target dimensions,
    and crops it to exact size (center crop).
    
    Args:
        image_url: URL of the image to download
        width: Target width in pixels
        height: Target height in pixels
    
    Returns:
        PIL Image object cropped to exact dimensions, or black fallback image
    """
    try:
        # Download image from URL
        r = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")

        # Calculate aspect ratios to determine resize strategy
        img_ratio = img.width / img.height
        target_ratio = width / height
        
         # Resize to fill the target dimensions (will be larger on one axis)
        if img_ratio > target_ratio:
            # Image is wider than target - match height
            new_height = height
            new_width = int(height * img_ratio)
        else:
            # Image is taller than target - match width
            new_width = width
            new_height = int(width / img_ratio)

         # Resize using high-quality Lanczos resampling
        img = img.resize((new_width, new_height), Image.LANCZOS)
        # Calculate crop coordinates to center the image
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        return img.crop((left, top, left + width, top + height))
    except:
        # If image loading fails, return a dark gray rectangle
        return Image.new("RGB", (width, height), (18, 18, 18))


# --------------------------------------------------
# WATERMARK
# --------------------------------------------------
def draw_branding(draw, width, height):
    """
    Draws a centered watermark/branding text at the bottom of the image.
    
    Args:
        draw: ImageDraw object to draw on
        width: Canvas width for centering
        height: Canvas height for positioning
    """
    text = "NEWS4YOU2026"
    font = get_font(26, bold=True)
     # Calculate text width for centering
    tw = draw.textbbox((0, 0), text, font=font)[2]
    # Draw centered text 60px from bottom
    draw.text(((width - tw) / 2, height - 60), text, fill=(180, 180, 180), font=font)


# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------
def calculate_text_height(draw, text, font, max_width, line_spacing):
     """
    Calculates how tall a text block will be when word-wrapped to fit max_width.
    
    Args:
        draw: ImageDraw object for text measurements
        text: Text string to measure
        font: Font to use for measurements
        max_width: Maximum width for wrapping
        line_spacing: Pixels between lines
    
    Returns:
        Tuple of (total_height, list_of_lines)
    """
    words = text.split()
    lines, line = [], ""
     # Word-wrap algorithm: add words until line exceeds max_width
    for w in words:
        test = f"{line} {w}".strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
             # Line is too long, save current line and start new one
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    # Calculate total height: sum of line heights + spacing between lines
    height = sum(font.getbbox(l)[3] for l in lines) + (len(lines) - 1) * line_spacing
    return height, lines


def draw_wrapped_text(draw, text, font, x, y, max_width, fill, line_spacing):
    """
    Draws word-wrapped text starting at (x, y).
    
    Args:
        draw: ImageDraw object
        text: Text to draw
        font: Font to use
        x, y: Starting coordinates
        max_width: Maximum width before wrapping
        fill: Text color
        line_spacing: Pixels between lines
    
    Returns:
        Y coordinate where text ended
    """
    # Get the wrapped lines
    _, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
    cy = y
    
     # Draw each line and advance Y position
    for l in lines:
        draw.text((x, cy), l, fill=fill, font=font)
        cy += font.getbbox(l)[3] + line_spacing
    return cy


def fit_text_in_box(draw, text, start_size, min_size, bold, max_width, max_height, line_spacing):
     """
    Finds the largest font size that fits text within the given dimensions.
    Tries sizes from start_size down to min_size in steps of 2.
    
    Args:
        draw: ImageDraw object
        text: Text to fit
        start_size: Starting font size to try
        min_size: Minimum acceptable font size
        bold: Whether to use bold font
        max_width: Maximum text width
        max_height: Maximum text height
        line_spacing: Pixels between lines
    
    Returns:
        Tuple of (font, list_of_lines, height)
    """
    size = start_size
    # Try progressively smaller sizes until text fits
    while size >= min_size:
        font = get_font(size, bold)
        h, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
        if h <= max_height:
            return font, lines, h
        size -= 2
    # If even minimum size doesn't fit, return it anyway
    font = get_font(min_size, bold)
    h, lines = calculate_text_height(draw, text, font, max_width, line_spacing)
    return font, lines, h


def split_text_into_slides(text, max_chars=400):
     """
    Splits long text into chunks of approximately max_chars length,
    breaking on word boundaries.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
    
    Returns:
        List of text chunks
    """
    words, slides, current = text.split(), [], []
    length = 0
    # Build slides word by word
    for w in words:
        if length + len(w) + 1 <= max_chars:
            # Word fits in current slide
            current.append(w)
            length += len(w) + 1
        else:
            # Word would exceed limit, start new slide
            slides.append(" ".join(current))
            current, length = [w], len(w)
    if current:
        # Don't forget the last slide
        slides.append(" ".join(current))
    return slides


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def generate_carousel(article, topic):
    """
    Generates a multi-slide Instagram/social media carousel from article data.
    First slide: Image with title overlay
    Subsequent slides: Text content on dark background
    
    Args:
        article: Dictionary with keys: 'image', 'title', 'desc', 'source'
        topic: Topic/category label for the article
    
    Returns:
        List of file paths to generated slide images
    """
    # Canvas dimensions (Instagram square format)
    WIDTH, HEIGHT = 1080, 1080
    margin_x = 80
    max_width = WIDTH - (2 * margin_x)

    # Reserve space for watermark at bottom
    WATERMARK_SPACE = 110

    # Initialize fonts for different text types
    subtitle_font = get_font(32)
    body_font = get_font(44)
    meta_font = get_font(30, bold=True)

    slide_paths = [] # Will store file paths of generated slides

    IMAGE_HEIGHT = int(HEIGHT * 0.6)  # First slide: 60% image, 40% text

    # ================= SLIDE 1 =================
    # Create black canvas
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    # Load and place background image in top portion
    bg = load_background(article.get("image"), WIDTH, IMAGE_HEIGHT)
    img.paste(bg, (0, 0))

    draw = ImageDraw.Draw(img)
    # Draw black rectangle for bottom text area
    draw.rectangle((0, IMAGE_HEIGHT, WIDTH, HEIGHT), fill=(0, 0, 0))

    # Topic bar configuration
    TOPIC_BAR_HEIGHT = 70
    # Black background for topic bar
    draw.rectangle((0, IMAGE_HEIGHT, WIDTH, IMAGE_HEIGHT + TOPIC_BAR_HEIGHT), fill=(0, 0, 0))
    # Subtle separator line above topic bar
    draw.rectangle((0, IMAGE_HEIGHT - 8, WIDTH, IMAGE_HEIGHT), fill=(15, 15, 15))

    # Add watermark branding
    draw_branding(draw, WIDTH, HEIGHT)

    # Maximum Y coordinate before watermark area
    MAX_TEXT_Y = HEIGHT - WATERMARK_SPACE

    # Fit title text, starting at 68pt and shrinking if needed
    title_font, title_lines, _ = fit_text_in_box(
        draw, article.get("title", ""),
        68, 42, True, max_width, 260, 14
    )

    # Prepare subtitle (truncated description)
    subtitle_text = article.get("desc", "")[:90] + "..."
    subtitle_h, _ = calculate_text_height(draw, subtitle_text, subtitle_font, max_width, 10)

    # Draw topic label in gold color, vertically centered in topic bar
    draw.text(
        (margin_x, IMAGE_HEIGHT + (TOPIC_BAR_HEIGHT - meta_font.size) // 2),
        topic.upper(),
        fill="#FFD700",
        font=meta_font
    )

    # Draw title lines, stopping if we run out of space
    y = IMAGE_HEIGHT + TOPIC_BAR_HEIGHT + 30
    for line in title_lines:
        lh = title_font.getbbox(line)[3]
        if y + lh > MAX_TEXT_Y:
            break
        draw.text((margin_x, y), line, fill="white", font=title_font)
        y += lh + 14

    # Draw subtitle if there's room
    if y + subtitle_h < MAX_TEXT_Y:
        draw_wrapped_text(draw, subtitle_text, subtitle_font, margin_x, y + 12, max_width, "#E0E0E0", 10)

    # Save first slide with timestamp to ensure unique filename
    p1 = f"slide_1_{int(time.time())}.png"
    img.save(p1, quality=95, optimize=True)
    slide_paths.append(p1)

    # ================= SLIDE 2+ =================
    # Split description into manageable chunks
    chunks = split_text_into_slides(article.get("desc", ""), 400)

    for i, chunk in enumerate(chunks):
        # Create dark gray canvas
        img = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 18))
        draw = ImageDraw.Draw(img)
        draw_branding(draw, WIDTH, HEIGHT)

        # Define safe areas (avoid top/bottom edges)
        TOP_SAFE = 160
        BOTTOM_SAFE = 160

        # Show topic and slide number if multiple slides
        label = f"{topic.upper()} ({i+1}/{len(chunks)})" if len(chunks) > 1 else topic.upper()
        draw.text((margin_x, 80), label, fill="#888888", font=meta_font)

        # Calculate vertical centering for content
        total_h, _ = calculate_text_height(draw, chunk, body_font, max_width, 22)
        usable_h = HEIGHT - TOP_SAFE - BOTTOM_SAFE
        centered_y = TOP_SAFE + (usable_h - total_h) // 2

        # Draw main content text, vertically centered
        draw_wrapped_text(draw, chunk, body_font, margin_x, centered_y, max_width, "white", 22)

        # Draw blue accent bar on left side
        draw.rectangle(
            (margin_x - 35, (HEIGHT - 500)//2, margin_x - 25, (HEIGHT + 500)//2),
            fill="#3aa0ff"
        )

        # On last slide, show source; otherwise show "continue" arrow
        if i == len(chunks) - 1:
            draw.text((margin_x, HEIGHT - 130), f"Source: {article.get('source', 'News')}", fill="#666666", font=meta_font)
        else:
            draw.text((WIDTH - 150, HEIGHT - 150), "→", fill="#3aa0ff", font=get_font(50, True))

        #Save content slide with unique filename
        p = f"slide_content_{i}_{int(time.time())}.png"
        img.save(p, quality=95, optimize=True)
        slide_paths.append(p)
        time.sleep(0.1) # Small delay to ensure unique timestamps

        #Limit to 10 slides maximum
        if len(slide_paths) >= 10:
            break

    return slide_paths
