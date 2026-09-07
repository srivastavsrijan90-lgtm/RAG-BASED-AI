import whisper
import json
import os

# Load Whisper model
model = whisper.load_model("small")

# Get all files in the audios folder
audios = os.listdir("audios")

for audio in audios:
    if "_" in audio:
        title = audio.split("_")[0]
        print(f"Processing: {title}")

        # Full path to the audio file
        audio_path = os.path.join("audios", audio)

        result = model.transcribe(
            audio=audio_path,
            language="hi",
            task="translate",
            word_timestamps=False
        )

        chunks = []

        for segment in result["segments"]:
            chunks.append({
                
                "title":title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {"chunks":chunks, "text":result["text"]}    

        # Save output using the title as filename
        with open(f"jsons/{audio}.json", "w")as f:
            json.dump(chunks_with_metadata,f) 