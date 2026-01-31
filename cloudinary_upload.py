import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="dciqxvk5p",
    api_key="165463927626715",
    api_secret="ef-FvlQpDTIalwIF8tmj5UX7faY",
    secure=True
)

def upload_image(image_path):
    result = cloudinary.uploader.upload(
        image_path,
        resource_type="image",
        folder="news-4-you"
    )
    return result["secure_url"]
