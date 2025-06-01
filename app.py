import streamlit as st
import re
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from googleapiclient.discovery import build
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",  # Atau ganti jadi None jika tidak mau tulis ke file
    filemode="a"          # Append log
)

# Load model & tokenizer dari Hugging Face Hub
@st.cache_resource
def load_model():
    logging.info("Memuat model dan tokenizer...")
    model_path = "Sadinal/fine_tuned_t5_indonesian_youtube_NLP"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Model berhasil dimuat ke device: {device}")
    return model.to(device), tokenizer, device

model, tokenizer, device = load_model()

# === Utility Functions ===
def get_youtube_comments(video_url, max_comments=20):
    logging.info("Mengekstrak ID video dari URL...")
    video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", video_url)
    if not video_id_match:
        logging.warning("URL YouTube tidak valid.")
        return []
    video_id = video_id_match.group(1)

    logging.info(f"Mengambil komentar untuk video ID: {video_id}")
    youtube = build('youtube', 'v3', developerKey=st.secrets["YOUTUBE_API_KEY"])
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= max_comments:
                    break
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except Exception as e:
        logging.error(f"Gagal mengambil komentar: {e}")
        return []
    
    logging.info(f"Jumlah komentar yang diambil: {len(comments)}")
    return comments

def preprocess_comments(comments):
    logging.info("Memproses dan membersihkan komentar...")
    cleaned_comments = []
    for c in comments:
        c = c.lower()
        c = re.sub(r'[^a-zA-Z0-9\s]', '', c)
        c = re.sub(r'\s+', ' ', c).strip()
        if c:
            cleaned_comments.append(c)
    logging.info(f"Komentar setelah dibersihkan: {len(cleaned_comments)}")
    return cleaned_comments

def join_comments_to_paragraph(cleaned_comments):
    logging.info("Menggabungkan komentar menjadi paragraf...")
    return ', '.join(cleaned_comments)

def generate_summary(text, model, tokenizer, max_input_length=512, max_output_length=100):
    logging.info("Menghasilkan ringkasan...")
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=max_input_length, truncation=True).to(model.device)
    summary_ids = model.generate(input_ids, max_length=max_output_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    logging.info("Ringkasan berhasil dibuat.")
    return summary

# === Streamlit UI ===
st.title("🧠 YouTube Comment Summarizer (Indonesian)")

video_url = st.text_input("📺 Masukkan URL Video YouTube:")
max_comments = st.slider("🔢 Jumlah Komentar", 5, 50, 20)

if st.button("🔍 Proses dan Ringkas"):
    with st.spinner("⏳ Memproses komentar..."):
        logging.info("Proses dimulai...")
        comments = get_youtube_comments(video_url, max_comments)

        if not comments:
            st.warning("⚠️ Tidak ada komentar yang ditemukan atau gagal mengambil komentar.")
            logging.warning("Komentar kosong atau gagal diambil.")
        else:
            cleaned = preprocess_comments(comments)
            if not cleaned:
                st.warning("❌ Komentar ditemukan, tetapi tidak valid setelah dibersihkan.")
                logging.warning("Komentar tidak valid setelah pembersihan.")
            else:
                paragraph = join_comments_to_paragraph(cleaned)
                summary = generate_summary(paragraph, model, tokenizer)

                st.subheader("💬 Contoh Komentar")
                for c in comments[:3]:
                    st.write("- " + c[:100])

                st.subheader("📝 Ringkasan Model")
                st.success(summary)
