import os
import math
import tempfile
from moviepy import VideoFileClip, AudioFileClip
import speech_recognition as sr
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import gradio as gr

# -----------------------------
# Load Pegasus model once
# -----------------------------

model_name = "google/pegasus-cnn_dailymail"

tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


# -----------------------------
# Helpers
# -----------------------------
def extract_audio(video_path):
    output_path = os.path.splitext(video_path)[0] + ".mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path, logger=None)
    clip.close()
    return output_path

def convert_and_transcribe(audio_mp3):
    recognizer = sr.Recognizer()
    # Convert mp3 to wav (required by SpeechRecognition)
    audio_wav = os.path.splitext(audio_mp3)[0] + ".wav"
    clip = AudioFileClip(audio_mp3)
    clip.write_audiofile(audio_wav, logger=None)
    clip.close()

    text_output = ""
    with sr.AudioFile(audio_wav) as source:
        total_duration = source.DURATION

    chunk_length = 30
    num_chunks = math.ceil(total_duration / chunk_length)

    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, total_duration)
        with sr.AudioFile(audio_wav) as src:
            audio_data = recognizer.record(src, offset=start, duration=end - start)
            try:
                chunk_text = recognizer.recognize_google(audio_data)
                text_output += chunk_text + " "
            except sr.UnknownValueError:
                # Chunk unclear — skipped
                pass
            except sr.RequestError:
                break

    return text_output.strip()

def summarize_text(text):
    if not text:
        return "No extracted text found."
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

def convert_to_bullets(summary):
    sentences = [s.strip() for s in summary.replace("<n>", ". ").split(".") if s.strip()]
    bullet_points = "\n".join([f"• {s}." for s in sentences])
    return bullet_points

def process_video(input_video):
    audio_path = extract_audio(input_video)
    extracted_text = convert_and_transcribe(audio_path)
    summary = summarize_text(extracted_text)
    bullet_summary = convert_to_bullets(summary)
    return extracted_text, bullet_summary


# -----------------------------
# Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload your video (MP4, MOV, WebM)"),

    outputs=[
        gr.Textbox(label="Transcribed text", lines=8),
        gr.Textbox(label="Bullet Summary", lines=8)
    ],
    title="Video Summarizer",
    description="Upload a video file — I'll extract audio, transcribe speech, and summarize into concise bullet points.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
