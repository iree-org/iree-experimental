import io
import requests
from PIL import Image


# Returns a sample image in the Imagenet2012 Validation Dataset.
def get_image_input(width=224, height=224):
    # We use an image of 5 applies since this is an easy example.
    img_path = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"
    data = requests.get(img_path).content
    img = Image.open(io.BytesIO(data))
    img = img.resize((width, height))
    return img
