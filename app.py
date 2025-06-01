import streamlit as st
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from googleapiclient.discovery import build
import torch

# Load model & tokenizer dari Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = "Sadinal/fine_tuned_t5_indonesian_youtube"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device

model, tokenizer, device = load_model()

# === Utility Functions ===
def get_youtube_comments(video_url, max_comments=20):
    video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", video_url)
    if not video_id_match:
        st.error("❌ Invalid YouTube URL")
        return []
    video_id = video_id_match.group(1)

    youtube = build('youtube', 'v3', developerKey=st.secrets["YOUTUBE_API_KEY"])
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            textFormat="plainText",
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

def preprocess_comments(comments):
    cleaned_comments = []
    for c in comments:
        c = c.lower()
        c = re.sub(r'[^a-zA-Z0-9\s]', '', c)
        c = re.sub(r'\s+', ' ', c).strip()
        if c:
            cleaned_comments.append(c)
    return cleaned_comments

def join_comments_to_paragraph(cleaned_comments):
    return ', '.join(cleaned_comments)

def generate_summary(text, model, tokenizer, max_input_length=512, max_output_length=100):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=max_input_length, truncation=True).to(model.device)
    summary_ids = model.generate(input_ids, max_length=max_output_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# === Streamlit UI ===
st.title("🧠 YouTube Comment Summarizer (Indonesian)")

video_url = st.text_input("📺 Masukkan URL Video YouTube:")
max_comments = st.slider("🔢 Jumlah Komentar", 5, 50, 20)

if st.button("🔍 Proses dan Ringkas"):
    with st.spinner("Mengambil komentar dan memproses..."):
        comments = get_youtube_comments(video_url, max_comments)
        if not comments:
            st.warning("Tidak ada komentar ditemukan.")
        else:
            cleaned = preprocess_comments(comments)
            paragraph = join_comments_to_paragraph(cleaned)
            summary = generate_summary(paragraph, model, tokenizer)

            st.subheader("💬 Contoh Komentar")
            for c in comments[:3]:
                st.write("- " + c[:100])

            st.subheader("📝 Ringkasan Model")
            st.success(summary)
