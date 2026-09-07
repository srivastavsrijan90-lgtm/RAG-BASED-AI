import os
import subprocess

files = os.listdir("videos")

print(files)

for file in files:
    print(file)

    filename = os.path.splitext(file)[0]
    print(filename)

    subprocess.run([
        "ffmpeg",
        "-i",
        f"videos/{file}",
        f"audios/{filename}.mp3"
    ]) 