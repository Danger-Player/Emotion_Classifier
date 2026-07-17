import os
import cloudinary
import cloudinary.uploader
from io import BytesIO


required_vars = ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"]
missing = [v for v in required_vars if not os.environ.get(v)]
if missing:
    raise RuntimeError(f"Missing required env vars: {missing}")


cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True
)


def upload_image(pil_image, public_id, folder="saved_moments"):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    try:
        result = cloudinary.uploader.upload(
            buffer,
            public_id=public_id,
            folder=folder
        )
        return result["secure_url"]
    except Exception as e:
        print(f"Cloudinary upload failed: {e}")
        return None