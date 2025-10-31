import os
import math
from flask import Flask, render_template, request
from moviepy import VideoFileClip, AudioFileClip
import speech_recognition as sr
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Load Pegasus model once
# -----------------------------
os.environ["HF_TOKEN"] = ""
model_name = "google/pegasus-cnn_dailymail"

print("🔹 Loading Pegasus model...")
tokenizer = PegasusTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
model = PegasusForConditionalGeneration.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))


# -----------------------------
# Helper: Extract Audio
# -----------------------------
def extract_audio(video_path):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "audio.mp3")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path)
    clip.close()
    return output_path


# -----------------------------
# Helper: Convert & Transcribe
# -----------------------------
def convert_and_transcribe(audio_mp3):
    recognizer = sr.Recognizer()
    audio_wav = os.path.join(app.config['UPLOAD_FOLDER'], "audio.wav")

    print("🎵 Converting MP3 to WAV using moviepy...")
    clip = AudioFileClip(audio_mp3)
    clip.write_audiofile(audio_wav)
    clip.close()

    text_output = ""
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
                print(f"❌ Google request error: {e}")
                break

    return text_output.strip()


# -----------------------------
# Helper: Summarize text
# -----------------------------
def summarize_text(input_file="static/uploads/extract.txt", output_file="static/uploads/summary.txt"):
    if not os.path.exists(input_file):
        return "❌ 'extract.txt' not found."

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return "⚠️ 'extract.txt' is empty — nothing to summarize."

    print("📄 Summarizing extracted text...")
    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")

    summary_ids = model.generate(
        **inputs,
        max_length=200,
        min_length=60,
        length_penalty=2.0,
        num_beams=5,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"💾 Summary saved to '{output_file}'")
    return summary


# -----------------------------
# Helper: Convert summary → bullet points
# -----------------------------
def convert_to_bullets(summary_file="static/uploads/summary.txt", output_file="static/uploads/bullet_summary.txt"):
    if not os.path.exists(summary_file):
        return "❌ 'summary.txt' not found."

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = f.read().strip()

    summary = summary.replace("<n>", ". ")
    sentences = [s.strip() for s in summary.split(".") if s.strip()]

    bullet_points = "\n".join([f"• {s}." for s in sentences])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(bullet_points)

    print(f"💾 Bullet-point summary saved to '{output_file}'")
    return bullet_points


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index2():
    extracted_text = None
    summary_text = None
    bullet_text = None

    if request.method == "POST":
        if "recorded" in request.files:
            file = request.files["recorded"]
            if file:
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded.webm")
                file.save(video_path)
                print("🎥 Video recorded and uploaded successfully!")

                # Step 1: Extract Audio
                audio_path = extract_audio(video_path)

                # Step 2: Transcribe
                extracted_text = convert_and_transcribe(audio_path)

                # Step 3: Save extracted text
                with open(os.path.join(app.config['UPLOAD_FOLDER'], "extract.txt"), "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                # Step 4: Summarize
                summary_text = summarize_text("static/uploads/extract.txt")

                # Step 5: Convert to bullet points
                bullet_text = convert_to_bullets("static/uploads/summary.txt")

    return render_template("index2.html", text=extracted_text, summary=summary_text, bullets=bullet_text)


if __name__ == "__main__":
    app.run(debug=True)
