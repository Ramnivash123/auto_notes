import os
import math
from flask import Flask, render_template, request, redirect, url_for
from moviepy import VideoFileClip, AudioFileClip
import speech_recognition as sr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def extract_audio(video_path):
    """Extracts audio from video as MP3"""
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "audio.mp3")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path)
    clip.close()
    return output_path


def convert_and_transcribe(audio_mp3):
    """Converts MP3 to WAV and extracts text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    audio_wav = os.path.join(app.config['UPLOAD_FOLDER'], "audio.wav")

    # Convert to WAV
    print("🎵 Converting MP3 to WAV using moviepy...")
    clip = AudioFileClip(audio_mp3)
    clip.write_audiofile(audio_wav)
    clip.close()

    text_output = ""
    print("🎙️ Splitting and converting speech to text...")

    # Get duration safely
    with sr.AudioFile(audio_wav) as source:
        total_duration = source.DURATION
        print(f"⏱️ Total duration: {total_duration:.2f} seconds")

    chunk_length = 30
    num_chunks = math.ceil(total_duration / chunk_length)

    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, total_duration)
        print(f"🧩 Processing chunk {i+1}/{num_chunks} ({start:.1f}-{end:.1f}s)")

        with sr.AudioFile(audio_wav) as src:
            audio_data = recognizer.record(src, offset=start, duration=end - start)
            try:
                chunk_text = recognizer.recognize_google(audio_data)
                text_output += chunk_text + " "
            except sr.UnknownValueError:
                print(f"⚠️ Chunk {i+1} unclear — skipped.")
            except sr.RequestError as e:
                print(f"❌ Google request error on chunk {i+1}: {e}")
                break

    return text_output.strip()


@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = None
    if request.method == "POST":
        file = request.files["video"]
        if file:
            # Save video
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)

            # Step 1: Extract audio
            audio_path = extract_audio(video_path)

            # Step 2: Convert & transcribe
            extracted_text = convert_and_transcribe(audio_path)

            # Step 3: Save to file
            with open(os.path.join(app.config['UPLOAD_FOLDER'], "extract.txt"), "w", encoding="utf-8") as f:
                f.write(extracted_text)

    return render_template("index.html", text=extracted_text)


if __name__ == "__main__":
    app.run(debug=True)
