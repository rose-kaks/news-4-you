# carousel_renderer.py

from PIL import Image, ImageDraw, ImageFont
import time


import requests
from io import BytesIO


def load_background(image_url, width=1080, height=1080):
    try:
        r = requests.get(image_url, timeout=10)
        bg = Image.open(BytesIO(r.content)).convert("RGB")
        return bg.resize((width, height))
    except:
        return Image.new("RGB", (width, height), (18, 18, 18))


def apply_smart_gradient(img):
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    start_y = int(height * 0.45)
    for y in range(start_y, height):
        alpha = int(235 * ((y - start_y) / (height - start_y)) ** 1.2)
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

    img = img.convert("RGBA")
    return Image.alpha_composite(img, overlay).convert("RGB")


def generate_carousel(article, topic):
    # uses your existing:
    # load_background
    # apply_smart_gradient
    # get_font
    # draw_wrapped_text
    # draw_branding

    WIDTH, HEIGHT = 1080, 1080
    paths = []

    # SLIDE 1
    img = apply_smart_gradient(load_background(article["image"]))
    draw = ImageDraw.Draw(img)
    draw_branding(draw, WIDTH, HEIGHT)

    draw_wrapped_text(
        draw,
        article["title"],
        get_font(68, True),
        80, 480,
        WIDTH - 160,
        "white",
        15
    )

    p1 = f"slide_1_{int(time.time())}.png"
    img.save(p1, quality=95)
    paths.append(p1)

    # SLIDE 2
    img = Image.new("RGB", (WIDTH, HEIGHT), (18,18,18))
    draw = ImageDraw.Draw(img)
    draw_branding(draw, WIDTH, HEIGHT)

    draw_wrapped_text(
        draw,
        article["desc"],
        get_font(44),
        80, 0,
        WIDTH - 160,
        "white",
        22,
        center_vertically=True
    )

    p2 = f"slide_2_{int(time.time())}.png"
    img.save(p2, quality=95)
    paths.append(p2)

    return paths
