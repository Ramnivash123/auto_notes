import os
import math
import streamlit as st
from moviepy import VideoFileClip, AudioFileClip
import speech_recognition as sr
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

st.set_page_config(page_title="Video-Summarizer", layout="wide")

# -----------------------------
# Setup paths
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Pegasus model once
# -----------------------------
os.environ["HF_TOKEN"] = ""
model_name = "google/pegasus-cnn_dailymail"

@st.cache_resource
def load_model():
    st.write("🔹 Loading Pegasus model...")
    tokenizer = PegasusTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    model = PegasusForConditionalGeneration.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    return tokenizer, model

tokenizer, model = load_model()


# -----------------------------
# Helper: Extract Audio
# -----------------------------
def extract_audio(video_path):
    output_path = os.path.join(UPLOAD_FOLDER, "audio.mp3")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path, logger=None)
    clip.close()
    return output_path


# -----------------------------
# Helper: Convert & Transcribe
# -----------------------------
def convert_and_transcribe(audio_mp3):
    recognizer = sr.Recognizer()
    audio_wav = os.path.join(UPLOAD_FOLDER, "audio.wav")

    st.info("🎵 Converting MP3 to WAV...")
    clip = AudioFileClip(audio_mp3)
    clip.write_audiofile(audio_wav, logger=None)
    clip.close()

    text_output = ""
    with sr.AudioFile(audio_wav) as source:
        total_duration = source.DURATION

    chunk_length = 30
    num_chunks = math.ceil(total_duration / chunk_length)

    st.info(f"🎙️ Processing {num_chunks} chunks...")

    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, total_duration)
        st.write(f"🧩 Chunk {i+1}/{num_chunks} ({start:.1f}-{end:.1f}s)")
        with sr.AudioFile(audio_wav) as src:
            audio_data = recognizer.record(src, offset=start, duration=end - start)
            try:
                chunk_text = recognizer.recognize_google(audio_data)
                text_output += chunk_text + " "
            except sr.UnknownValueError:
                st.warning(f"⚠️ Chunk {i+1} unclear — skipped.")
            except sr.RequestError as e:
                st.error(f"❌ Google API error: {e}")
                break

    return text_output.strip()


# -----------------------------
# Helper: Summarize text
# -----------------------------
def summarize_text(text):
    if not text:
        return "⚠️ No extracted text found."

    st.info("📄 Summarizing extracted text...")
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
    return summary


# -----------------------------
# Helper: Convert summary → bullet points
# -----------------------------
def convert_to_bullets(summary):
    sentences = [s.strip() for s in summary.replace("<n>", ". ").split(".") if s.strip()]
    bullet_points = "\n".join([f"• {s}." for s in sentences])
    return bullet_points


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Video Summarizer")
st.markdown("Upload a video file — I’ll extract audio, transcribe speech, and summarize it into concise bullet points.")

uploaded_file = st.file_uploader("Upload your video (MP4, MOV, or WebM)", type=["mp4", "mov", "webm"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    if st.button("▶️ Process Video"):
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio(video_path)
        with st.spinner("Transcribing speech..."):
            extracted_text = convert_and_transcribe(audio_path)

        st.subheader("📝 Extracted Text")
        st.text_area("Transcribed text", extracted_text, height=200)

        with st.spinner("Summarizing..."):
            summary = summarize_text(extracted_text)
            bullet_summary = convert_to_bullets(summary)

        st.subheader("🧠 Summary")
        st.text_area("Bullet Summary", bullet_summary, height=200)

        # Download option
        st.download_button(
            label="⬇️ Download Bullet Summary as TXT",
            data=bullet_summary,
            file_name="bullet_summary.txt",
            mime="text/plain"
        )
else:
    st.info("Please upload a video file to start.")
