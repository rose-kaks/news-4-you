import os
import cloudinary
import cloudinary.uploader

#--------------------------------------------------
# CLOUDINARY CONFIGURATION
# --------------------------------------------------
# Configure Cloudinary with credentials from environment variables
# This keeps sensitive API keys out of the source code
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"), # Your Cloudinary account name
    api_key=os.getenv("CLOUDINARY_API_KEY"), # API key for authentication
    api_secret=os.getenv("CLOUDINARY_API_SECRET"), # API secret for authentication
    secure=True # Use HTTPS for all uploads and URLs
)

# --------------------------------------------------
# IMAGE UPLOAD FUNCTION
# --------------------------------------------------

def upload_image(image_path):
    """
    Uploads an image file to Cloudinary cloud storage.
    
    Args:
        image_path: Local file path to the image to upload
    
    Returns:
        String containing the secure HTTPS URL of the uploaded image
        (can be used to access the image from anywhere on the web)
    """
    # Upload the image to Cloudinary
    result = cloudinary.uploader.upload(
        image_path, # Path to local file
        resource_type="image", # Specify this is an image (not video/raw file)
        folder="news-4-you" # Organize uploads into a folder on Cloudinary
    )

    # Extract and return the secure URL from the response
    # This URL can be embedded in web pages, shared, etc.
    return result["secure_url"]
