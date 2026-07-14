import os
import pandas as pd

df = pd.read_json("/home/administrator/.cache/kagglehub/datasets/crawlfeeds/images-extracted-from-fashion-website/versions/1/farfetch_fashion_dataset_images_crawlfeeds.json")


df["image_file"] = df["image_file"].str.split("|").str[0].str.strip()

base_path = "/home/administrator/.cache/kagglehub/datasets/crawlfeeds/images-extracted-from-fashion-website/versions/1/images"
df["absolute_path"] = df["image_file"].apply(lambda x : os.path.join(base_path, x))

essential_columns = ["absolute_path", "uniq_id", "brand", "title", "url"]
df_clean = df[essential_columns]


try:
    os.mkdir("data")
except OSError as e:
    print("Error: ", e)
df_clean.to_csv("data/catalog.csv", index=False)
