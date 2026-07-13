import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


path = "/home/administrator/.cache/kagglehub/datasets/crawlfeeds/images-extracted-from-fashion-website/versions/1/images/images/10851775_0.jpg"
image = Image.open(path)


inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

pooler_output = image_features.pooler_output
print(pooler_output.shape)