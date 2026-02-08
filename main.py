#--------------------------------------------------
# IMPORTS
# --------------------------------------------------
# Import all the necessary functions from other module
from carousel_renderer import generate_carousel  # Creates Instagram carousel slide images
from cloudinary_upload import upload_image # Uploads images to Cloudinary cloud storage
from insta_publish import post_carousel  # Publishes carousel to Instagram
from news_pipeline import get_next_article, load_db, save_db, mark_posted # News article management


# --------------------------------------------------
# MAIN ORCHESTRATION FUNCTION
# --------------------------------------------------
def main():
     """
    Main workflow to automatically post news articles as Instagram carousels.
    
    Complete pipeline:
    1. Get next unposted article from the news database
    2. Generate carousel slide images from the article
    3. Upload slides to Cloudinary for public hosting
    4. Post the carousel to Instagram with a caption
    5. Mark article as posted in the database to avoid duplicates
    """

    # --------------------------------------------------
    # STEP 1: GET NEXT ARTICLE
    # --------------------------------------------------
    # Retrieve the next article that hasn't been posted yet
    # Returns None if no articles are available or if topic cooldown is active
    result = get_next_article()

    if not result:
        print("😴 No fresh topics found (Topic Cooldown active).")
        return
    # Extract the article data from the result
    article = result["article"]
    # Get the topic/category for this article
    # topic = article.get("entities", ["General"])[0]
    topic=article["topic"]

    # --------------------------------------------------
    # STEP 2: GENERATE CAROUSEL IMAGES
    # --------------------------------------------------
    # 1. Generate carousel images
    # Creates multiple PNG files (slides) from the article content
    # Returns a list of local file paths
    slide_paths = generate_carousel(article, topic)

    # --------------------------------------------------
    # STEP 3: UPLOAD TO CLOUDINARY
    # --------------------------------------------------
    # 2. Upload images to Cloudinary
    # Instagram requires publicly accessible URLs, so we upload to cloud storage
    public_urls = []
    for image_path in slide_paths:
        # Upload each slide and get back a public HTTPS URL
        url = upload_image(image_path)
        public_urls.append(url)

    # Log all the public URLs for debugging/verification
    print("🌍 Public image URLs:")
    for u in public_urls:
        print(u)

    # --------------------------------------------------
    # STEP 4: POST TO INSTAGRAM
    # --------------------------------------------------
    # 3. Publish to Instagram
    # Create a caption with article title, source, and hashtags
    caption = f"""{article['title']}

Source: {article.get('source')}
#news #technology #india
"""
    # Attempt to post the carousel to Instagram
    # Returns True if successful, False if it failed
    success = post_carousel(public_urls, caption)

     # --------------------------------------------------
    # STEP 5: UPDATE DATABASE
    # --------------------------------------------------
    # If posting succeeded, mark the article as posted in the database
    # This prevents the same article from being posted again
    if success:
        print("📤 Successfully posted to Instagram")

        db = load_db()  # Load the current database
        mark_posted(db, article) # Mark this article as posted
        save_db(db) # Save the updated database
    else:
        # If posting failed, don't mark as posted so it can be retried later
        print("❌ Instagram post failed. Article kept in queue.")


#--------------------------------------------------
# SCRIPT ENTRY POINT
# --------------------------------------------------
# This ensures main() only runs when the script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
