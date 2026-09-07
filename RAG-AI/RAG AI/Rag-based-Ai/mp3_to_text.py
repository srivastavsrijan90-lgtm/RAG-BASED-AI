import whisper
import json

model = whisper.load_model("small")

result = model.transcribe(
    audio="audios/sample.mp3",
    language="hi",
    task="translate",
    word_timestamps=False
)



chunks = []

for segment in result["segments"]:
    chunks.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"].strip()
    })

print(chunks)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)