import streamlit as st
import torch
import time
from PIL import Image
from embed import initialize_models, open_image, extract_image_embedding
from st_copy import copy_button
from pathlib import Path

@st.cache_resource
def load_model():
    return initialize_models()

def extract_text_embedding(text, device, model, processor):
    inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs).pooler_output[0].cpu().tolist()

    return text_embeddings

def get_results(collection, embeddings):
    results = collection.query(
            query_embeddings=[embeddings],
            n_results=4,
        )

    return results

def load_image(relative_path):
    base_dir = Path(__file__).resolve().parent
    local_path = base_dir / relative_path
    return local_path

def render_ui(): 
    st.title("Archive Recommender")

    uploaded_file = st.file_uploader("Upload a picture of clothing", type=["jpeg", "jpg", "png"])
    text = st.text_input("Enter a name of a brand, a specific piece of clothing, or describe a piece of clothing", max_chars=500)
    find_recommendations = st.button("Get recommendations")

    return uploaded_file, text, find_recommendations

def execute_similar_search(data, search_type, device, model, processor, collection, state_key):
    with st.spinner("Finding the best matches...", show_time=True):
        if search_type == "image":
            image = open_image(data)
            embeddings = extract_image_embedding(image, device, model, processor)
        elif search_type == "text":
            embeddings = extract_text_embedding(data, device, model, processor)

        similar_clothing_results = get_results(collection, embeddings)
        st.session_state[state_key] = similar_clothing_results

def show_results(results, header_text):
    st.header(header_text)
    cols = st.columns(4)

    metadatas = results["metadatas"][0]
    for i, metadata in enumerate(metadatas):
        with cols[i % 4]:
            st.image(load_image(metadata["images"]), caption=f"{metadata["brand"]} {metadata["title"]}")
    

# collection, device, model, processor = load_model()

# st.title("Archive Recommender")

# uploaded_file = st.file_uploader("Upload a picture of clothing", type=["jpeg", "jpg", "png"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)

#     with st.spinner("Finding the best matches...", show_time=True):
#         time.sleep(2)
#         # inputs = processor(images=image, return_tensors="pt").to(device)
#         # with torch.no_grad():
#         #     image_embeddings = model.get_image_features(**inputs).pooler_output[0].cpu().tolist()
#         image_embeddings = extract_image_embedding(uploaded_file, device, model, processor)

#         results = collection.query(
#             query_embeddings=[image_embeddings],
#             n_results=4
#         )
#         st.write(results)

#         st.header("Recommended brands with clothing items")
#         cols = st.columns(4)

#         metadatas = results["metadatas"][0]
#         for i, metadata in enumerate(metadatas):
#             with cols[i % 4]:
#                 st.image(load_image(metadata["images"]), caption=f"{metadata["brand"]} {metadata["title"]}")
                

# text = st.text_input("Enter a name of a brand, a specific piece of clothing, or describe a piece of clothing", max_chars=500)

# if st.button("Search") and text_input:
#     with st.spinner("Finding the best matches...", show_time=True):
#         time.sleep(2)
#         inputs = processor(text=text_input, return_tensors="pt").to(device)
#         with torch.no_grad():
#             text_embeddings = model.get_text_features(**inputs).pooler_output[0].cpu().tolist()

#         results = collection.query(
#             query_embeddings=[text_embeddings],
#             n_results=4,
#         )

#         st.header("Recommended Brands with Clothing Items")
#         cols = st.columns(4)
        
#         # Extracts the metadata relevevant to the query_embeddings. Enters the metadata list 
#         # that contains 4 dictionaries of metadata for the 4 closest neighbors. The loop iterates
#         # through the list of 4 dictionaries and writes the title and brand for the 4 closest neighbors.
#         metadatas = results["metadatas"][0]
#         for i, metadata in enumerate(metadatas):
#             # with col1:
#             #     st.image(load_image(metadata["images"]), width="stretch", caption=f"{metadata['brand']} {metadata['title']}")
#             #     #st.write(f"{metadata['brand']} {metadata['title']}")
#             #     # brand_item_text = f"{metadata["brand"]} {metadata["title"]}"

#             #     # copy_button(
#             #     #     brand_item_text,
#             #     #     icon=":material/content_copy:",
#             #     #     tooltip="Copy",
#             #     #     copied_label="copy_{item_id}",
#             #     # )
#             # with col2:
#             #     st.write("")
#             #     #st.image(load_image(metadata["images"]))
#             with cols[i % 4]:
#                 st.image(load_image(metadata["images"]), width="stretch", caption=f"{metadata['brand']} {metadata['title']}")

def main():
    try:
        if "similar_image_results" not in st.session_state:
            st.session_state["similar_image_results"] = None
        if "similar_text_results" not in st.session_state:
                st.session_state["similar_text_results"] = None
        
        collection, device, model, processor = load_model()
        uploaded_file, text, find_recommendations = render_ui()

        if uploaded_file:
            execute_similar_search(uploaded_file, "image", device, model, processor, collection, "similar_image_results")

        if find_recommendations and text:
            execute_similar_search(text, "text", device, model, processor, collection, "similar_text_results")


        if st.session_state["similar_image_results"]:
            show_results(st.session_state["similar_image_results"], "Recommended Items by Image Search")

        if st.session_state["similar_text_results"]:
            show_results(st.session_state["similar_text_results"], "Recommended Items by Text Search")
    except Exception as e:
        print("An error occured while searching through the archive...")




    # st.write(st.session_state["similar_image_results"])

if __name__ == "__main__":
    main()

    
 






