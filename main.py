from news_pipeline import get_next_article
from carousel_renderer import generate_carousel
from cloudinary_upload import upload_image
from insta_publish import post_carousel


def main():
    result = get_next_article()

    if not result:
        print("😴 No fresh topics found (Topic Cooldown active).")
        return

    article = result["article"]
    # topic = article.get("entities", ["General"])[0]
    topic=article["topic"]

    # 1. Generate carousel images
    slide_paths = generate_carousel(article, topic)

    # 2. Upload images to Cloudinary
    public_urls = []
    for image_path in slide_paths:
        url = upload_image(image_path)
        public_urls.append(url)

    print("🌍 Public image URLs:")
    for u in public_urls:
        print(u)

    # 3. Publish to Instagram
    caption = f"""{article['title']}

Source: {article.get('source')}
#news #technology #india
"""

    post_carousel(public_urls, caption)
    print("📤 Successfully posted to Instagram")


if __name__ == "__main__":
    main()
