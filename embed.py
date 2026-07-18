import os
import torch
import pandas as pd
import chromadb
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

load_dotenv()

def read_file(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["absolute_path"])
    return df.to_dict("records")

def initialize_models():
    client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
    )
    collection = client.get_or_create_collection(name="images_collection")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return collection, device, model, processor

def open_image(image_path):
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: file not found at {image_path}, continuing...")
        return None
    except UnidentifiedImageError:
        print(f"Error: image at {image_path} cannot be opened and identified, continuing...")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}, continuing...")
        return None

    return image

def extract_image_embedding(image, device, model, processor):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).pooler_output[0].cpu().tolist()

    return image_features

def upload_chunk_to_chroma(collection, ids, embeddings, metadata):
    collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadata,
    )

def run_extraction(catalog, collection, device, model, processor):
    ids_list = []
    embeddings_list = []
    metadata_list = []

    chunk_size = 250

    for row in catalog:
        path = row["absolute_path"]
        image = open_image(path)
        image_embeddings = extract_image_embedding(image, device, model, processor)

        if image_embeddings is None:
            continue

        embeddings_list.append(image_embeddings)
        metadata_list.append({"brand" : row["brand"], "title": row["title"]})
        ids_list.append(str(row["uniq_id"]))

        if len(ids_list) >= chunk_size:
            upload_chunk_to_chroma(collection, ids_list, embeddings_list, metadata_list)
            ids_list, embeddings_list, metadata_list = [], [], []

    if len(ids_list) > 0: 
        upload_chunk_to_chroma(collection, ids_list, embeddings_list, metadata_list)
            

def main():
    collection, device, model, processor = initialize_models()
    catalog = read_file("data/catalog.csv")
    run_extraction(catalog, collection, device, model, processor)

if __name__ == "__main__":
    main()

