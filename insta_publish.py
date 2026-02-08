# insta_publish.py

import os
import requests
import time

#--------------------------------------------------
# INSTAGRAM API CONFIGURATION
# --------------------------------------------------
# Load Instagram credentials from environment variables
INSTAGRAM_USER_ID = os.getenv("INSTAGRAM_USER_ID") # Your Instagram Business Account ID
ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN") # Facebook Graph API access token
GRAPH_API = "https://graph.facebook.com/v19.0" # Facebook Graph API base URL 

#--------------------------------------------------
# CREATE INDIVIDUAL IMAGE CONTAINER
# --------------------------------------------------
def create_image_container(image_url, caption=None):
    """
    Creates a media container for a single image that will be part of a carousel.
    This is the first step in posting to Instagram - containers must be created
    before they can be published.
    
    Args:
        image_url: Publicly accessible URL of the image (must be HTTPS)
        caption: Optional caption text (only used for single images, not carousel items)
    
    Returns:
        Container ID string if successful, None if all attempts fail
    """
    # Prepare the API request payload
    payload = {
        "image_url": image_url, # URL where Instagram can fetch the image
        "is_carousel_item": True, # Mark this as part of a carousel post
        "access_token": ACCESS_TOKEN # Authentication token
    }
    if caption:
        payload["caption"] = caption

    # Try 3 times because of the 'is_transient' nature of Code 2
    # Sometimes Instagram needs time to fetch and process the image from the URL
    for attempt in range(3):
        r = requests.post(f"{GRAPH_API}/{INSTAGRAM_USER_ID}/media", data=payload)
        data = r.json()

        # If we got an ID back, the container was created successfully
        if "id" in data:
            return data["id"]
        # Log the failure and try again
        print(f"⚠️ Attempt {attempt+1} failed for {image_url}: {data}")
        time.sleep(5) # Wait for the hosting service to fully propagate the image
        
    return None

# --------------------------------------------------
# WAIT FOR CONTAINER TO BE READY
# --------------------------------------------------
def wait_until_ready(creation_id, timeout=120):
    """
    Polls Instagram's API to check if a media container has finished processing.
    Instagram needs time to download, validate, and process uploaded media.
    
    Args:
        creation_id: The container ID returned from create_image_container
        timeout: Maximum seconds to wait (default 120 = 2 minutes)
    
    Returns:
        True if container is ready, False if it failed or timed out
    """
    start = time.time()
    # Keep checking until timeout is reached
    while time.time() - start < timeout:
        # Query the container's status
        r = requests.get(
            f"{GRAPH_API}/{creation_id}",
            params={
                "fields": "status_code", # We only need the status field
                "access_token": ACCESS_TOKEN
            }
        )
        res = r.json()
        status = res.get("status_code")

        # Container is ready to be published
        if status == "FINISHED":
            return True
        # Container processing failed
        elif status== "ERROR":
            print(f" Container {creation_id} failed: {res.get('error_description')}")
            return False

        # Still processing - wait and check again
        print(f"{creation_id} still {status}... waiting 10s")
        time.sleep(10)
    # Timed out waiting for container to be ready
    return False

# --------------------------------------------------
# CREATE CAROUSEL CONTAINER
# --------------------------------------------------
def create_carousel_container(children_ids, caption):
    """
    Creates a carousel post container that groups multiple image containers together.
    This is the second step after creating individual image containers.
    
    Args:
        children_ids: List of container IDs from create_image_container calls
        caption: Caption text for the entire carousel post
    
    Returns:
        Carousel container ID if successful, None if failed
    """
    payload = {
        "media_type": "CAROUSEL", # Specify this is a multi-image carousel
        "children": children_ids, # Pass as a list, not a joined string
        "caption": caption, # The text that appears with the post
        "access_token": ACCESS_TOKEN
    }
    # Send as JSON (not form data) for carousel creation
    r = requests.post(
        f"{GRAPH_API}/{INSTAGRAM_USER_ID}/media",
        json=payload
    )
    data = r.json()
    print("INSTAGRAM RESPONSE:", data)
    return data.get("id")

#--------------------------------------------------
# PUBLISH CONTAINER TO INSTAGRAM
# --------------------------------------------------
def publish_container(creation_id, retries=3):
    """
    Publishes a finalized media container to Instagram, making it visible on the profile.
    This is the final step that actually posts the content.
    
    Args:
        creation_id: The carousel container ID to publish
        retries: Number of retry attempts if publishing fails
    
    Returns:
        API response JSON if successful, None if all retries failed
    """
    for i in range(retries):
        r = requests.post(
            f"{GRAPH_API}/{INSTAGRAM_USER_ID}/media_publish",
            data={
                "creation_id": creation_id,
                "access_token": ACCESS_TOKEN
            }
        )
        # Publish succeeded
        if r.ok:
            return r.json()
        # Publish failed - log and retry
        print(f"⚠️ Publish attempt {i+1} failed:", r.text)
        # If it's a "Media not ready" error, wait longer
        # If it's a "Rate limit" error, wait MUCH longer
        wait_time = (i + 1) * 40 
        print(f"💤 Sleeping {wait_time}s before retrying...")
        time.sleep(wait_time)

    return None


# --------------------------------------------------
# MAIN FUNCTION: POST COMPLETE CAROUSEL
# --------------------------------------------------
def post_carousel(image_urls, caption):
    """
    Complete workflow to post a carousel of images to Instagram.
    
    Process:
    1. Create individual containers for each image
    2. Wait for all containers to finish processing
    3. Create a carousel container grouping them together
    4. Wait for carousel to finalize
    5. Publish the carousel to Instagram
    
    Args:
        image_urls: List of publicly accessible image URLs (HTTPS)
        caption: Text caption for the carousel post
    
    Returns:
        True if post was successful, False if it failed
    """
    # Instagram carousel limit is 10 slides
    if len(image_urls) > 10:
        print(f"⚠️ Warning: {len(image_urls)} slides provided. Capping at 10.")
        image_urls = image_urls[:10]
        
    # Instagram's minimum is 2
    if len(image_urls) < 2:
        print("❌ Not enough slides for a carousel.")
        return False
    
    child_ids = []

    # 1️⃣ create child containers
    # Create a separate container for each image in the carousel
    for url in image_urls:
        cid = create_image_container(url)
        if not cid:
            raise Exception("Failed to create image container")
        child_ids.append(cid)

    # 2️⃣ wait until all are ready
    # Instagram needs time to download and process each image
    for cid in child_ids:
        ok = wait_until_ready(cid)
        if not ok:
            print("⚠️ Child container not ready, continuing anyway")

    # 3️⃣ create carousel container
    # Group all the individual image containers into one carousel
    carousel_id = create_carousel_container(child_ids, caption)
    if not carousel_id:
        raise Exception("Failed to create carousel container")

    # 4. NEW: Wait for the CAROUSEL itself to be ready
    # This is likely why you got the 9007 error!
    # The carousel container also needs processing time before it can be published
    print("⏳ Waiting for carousel to finalize...")
    if not wait_until_ready(carousel_id):
        print("❌ Carousel container failed to finalize.")
        return False
    # 4️⃣ publish (SAFE)
    # Actually post the carousel to Instagram, making it visible
    result = publish_container(carousel_id)

    if not result:
        return False

    return True
