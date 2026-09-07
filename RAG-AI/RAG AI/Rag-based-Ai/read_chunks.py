# preprocess jsons

import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


def create_embeddings(texts):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": texts
        }
    )

    if response.status_code != 200:
        print("Ollama Error:", response.text)
        print("Number of texts:", len(texts))
        print("First text:", texts[0][:500] if texts else "EMPTY")
        response.raise_for_status()

    data = response.json()
    return data["embeddings"]

newjsons = os.listdir("newjsons")

my_dict = []
chunk_id = 0

for json_file in newjsons:
    with open(os.path.join("newjsons", json_file), "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content["chunks"]

    
    texts = [chunk["text"] for chunk in chunks]

  
    embeddings = create_embeddings(texts)

   
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} embeddings but got {len(embeddings)}"
        )

   
    for chunk, embedding in zip(chunks, embeddings):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embedding
        my_dict.append(chunk)
        chunk_id += 1

    print(f"Processed {json_file}")

print(f"Total chunks: {len(my_dict)}")

df = pd.DataFrame.from_records(my_dict)

joblib.dump(df, "embedding.joblib")




