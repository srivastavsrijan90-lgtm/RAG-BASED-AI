import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os
import json


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VIDEO_DIR = os.path.join(
    BASE_DIR,
    "Videos"
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": texts
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"]


# ============================================================
# OLLAMA INFERENCE
# ============================================================

def inference(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 250,
                "temperature": 0
            }
        },
        stream=True
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if line:

            data = json.loads(
                line.decode("utf-8")
            )

            text = data.get(
                "response",
                ""
            )

            if text:
                yield text


# ============================================================
# FORMAT TIMESTAMP
# ============================================================

def format_timestamp(seconds):

    minutes = int(seconds // 60)

    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:05.2f}"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

df = joblib.load(
    os.path.join(
        BASE_DIR,
        "embedding.joblib"
    )
)



# ============================================================
# PRELOAD EMBEDDINGS
# ============================================================

EMBEDDINGS = np.vstack(
    df["embedding"].values
)

# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def run_rag(incoming_query):

    # --------------------------------------------------------
    # QUESTION EMBEDDING
    # --------------------------------------------------------

    question_embedding = create_embeddings(
        [incoming_query]
    )[0]


    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    similarity = cosine_similarity(
    EMBEDDINGS,
    [question_embedding]
    ).flatten()


    # --------------------------------------------------------
    # TOP 3 RESULTS
    # --------------------------------------------------------

    top_result = 3

    max_indx = similarity.argsort()[
        ::-1
    ][:top_result]


    new_df = df.iloc[
        max_indx
    ].copy()


    # --------------------------------------------------------
    # CREATE TIMESTAMPS
    # --------------------------------------------------------

    new_df["start_time"] = new_df[
        "start"
    ].apply(
        format_timestamp
    )


    new_df["end_time"] = new_df[
        "end"
    ].apply(
        format_timestamp
    )


    # --------------------------------------------------------
    # CALCULATE DURATION
    # --------------------------------------------------------

    new_df["duration"] = (
        new_df["end"] -
        new_df["start"]
    ).apply(
        lambda x:
        f"{x:.2f} seconds"
    )


    # --------------------------------------------------------
    # VIDEO CHUNKS
    # --------------------------------------------------------

    video_chunks = new_df[
        [
            "title",
            "start_time",
            "end_time",
            "duration",
            "text"
        ]
    ].to_json(
        orient="records"
    )


    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a helpful course video assistant.

Use the retrieved video chunks below to answer the student's question.

VIDEO CHUNKS:
{video_chunks}

QUESTION:
{incoming_query}

RULES:
- Answer using ONLY the provided video chunks.
- Give a detailed but easy-to-understand answer.
- Explain all useful information available in the retrieved chunks.
- Do not invent information.
- Do not use outside knowledge.
- You can write multiple paragraphs.
- You can use bullet points or examples if they are present in the video chunks.
- Select the most relevant video.
- Copy the video title exactly.
- Copy start_time exactly.
- Copy end_time exactly.
- Copy duration exactly.
- Do not modify timestamps.
- Do not create a video title.
- Return only one recommended video.

FORMAT:

Topic: <topic>

Answer:
<detailed explanation based on the retrieved video content>

Video: <exact video title>

Timestamp: <exact start_time> - <exact end_time>

Duration: <exact duration>

What is taught:
<detailed explanation of what is available in the video chunk>

Where to start:
Start the video at <exact start_time>.
"""

    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    return inference(
        prompt
    )


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    incoming_query = input("ASK A QUESTION: ")

    print("=" * 70)
    print("RAG RESPONSE")
    print("=" * 70)

    full_response = ""

    for chunk in run_rag(incoming_query):

        # सिर्फ नया chunk print होगा
        print(
            chunk,
            end="",
            flush=True
        )

        # पूरा response memory में save होगा
        full_response += chunk

    print("\n")
    print("=" * 70)

    # Save final response
    with open(
        os.path.join(BASE_DIR, "responses.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(full_response)