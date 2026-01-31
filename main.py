# main.py

import os
import shutil

from news_pipeline import get_next_article
from carousel_renderer import generate_carousel
from insta_publish import post_carousel


# GitHub automatically provides this in Actions
# format: username/repo-name
repo_full = os.getenv("GITHUB_REPOSITORY")
if not repo_full:
    raise RuntimeError("GITHUB_REPOSITORY not found")

username, repo_name = repo_full.split("/")

PUBLIC_BASE = f"https://{username}.github.io/{repo_name}"
PUBLIC_SLIDES_DIR = "docs/slides"

os.makedirs(PUBLIC_SLIDES_DIR, exist_ok=True)


def main():
    result = get_next_article()

    if not result:
        print("😴 No article selected")
        return

    article = result["article"]
    topic = article.get("entities", ["General"])[0]

    # 1. Generate carousel images
    slide_paths = generate_carousel(article, topic)

    # 2. Move images to GitHub Pages directory
    public_urls = []
    for path in slide_paths:
        filename = os.path.basename(path)
        dest = os.path.join(PUBLIC_SLIDES_DIR, filename)
        shutil.move(path, dest)

        from cloudinary_upload import upload_image

        public_urls = []

        for image_path in generated_images:
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
