import os
import math
import json

# Har 5 chunks ko merge karna hai
N = 10

# Input aur output folders
INPUT_FOLDER = "jsons"
OUTPUT_FOLDER = "newjsons"

# Output folder create karo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Input folder check
if not os.path.exists(INPUT_FOLDER):
    print(f"ERROR: '{INPUT_FOLDER}' folder nahi mila.")
    print(f"Current folder: {os.getcwd()}")
    exit()

files_found = 0

for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith(".json"):
        continue

    files_found += 1

    input_path = os.path.join(INPUT_FOLDER, filename)

    try:
        # JSON file read karo
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])

        if not chunks:
            print(f"SKIP: {filename} main chunks are missing .")
            continue

        num_chunks = len(chunks)
        num_groups = math.ceil(num_chunks / N)

        new_chunks = []

        # Chunks ko groups mein merge karo
        for i in range(num_groups):

            start_idx = i * N
            end_idx = min((i + 1) * N, num_chunks)

            chunk_group = chunks[start_idx:end_idx]

            merged_chunk = {
                "number": chunk_group[0].get("number"),
                "title": chunk_group[0].get("title", ""),
                "start": chunk_group[0].get("start"),
                "end": chunk_group[-1].get("end"),
                "text": " ".join(
                    chunk.get("text", "")
                    for chunk in chunk_group
                )
            }

            new_chunks.append(merged_chunk)

        # Output file path
        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        # Merged JSON save karo
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": new_chunks,
                    "text": data.get("text", "")
                },
                f,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"DONE: {filename} | "
            f"{num_chunks} chunks -> {len(new_chunks)} chunks"
        )

    except Exception as e:
        print(f"ERROR in {filename}: {e}")


print()
print("=" * 50)
print(f"Total JSON files found: {files_found}")
print(f"Output folder: {OUTPUT_FOLDER}")
print("=" * 50)