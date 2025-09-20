import streamlit as st
from pathlib import Path
import tempfile
import cv2
from ultralytics import YOLO
import time

st.set_page_config(page_title="AquaFun", layout="wide")

# ====== LOAD CSS ======
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ====== SIDEBAR CONTROL ======
st.sidebar.header("⚙️ Video Settings")
scale = st.sidebar.slider("Scale (resize)", 0.1, 1.0, 0.5, 0.05)
skip = st.sidebar.slider("Skip frames", 0, 5, 1, 1)
conf_thres = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)

# ====== HEADER ======
st.markdown("""
<div class="header">
    <div class="logo">AquaFun</div>
    <div class="nav">
        <a href="#home">Home</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ====== HERO SECTION ======
st.markdown("""
<div class="hero" id="home">
    <h1>Welcome to AquaFun</h1>
    <p>Discover the fascinating world of fish — from habitats, behaviors, and ecological traits — right from your aquarium videos.</p>
</div>
""", unsafe_allow_html=True)

# ====== UPLOAD SECTION ======
st.markdown("<h2 class='upload-title'>🎥 Share your aquarium video with us!</h2>", unsafe_allow_html=True)
st.markdown("<p class='upload-subtitle'>Upload your aquarium video and watch detection happen live!</p>", unsafe_allow_html=True)

uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi", "mkv", "webm", "wmv", "mpeg"])

# ====== LOAD YOLO MODEL ======
model = YOLO("best.pt")  # pastikan file ada di folder proyek

# ====== SIMULASI REAL-TIME ======
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Hilangkan delay penuh supaya lebih cepat
    delay = 1 / fps if fps > 0 else 0

    frame_id = 0
    annotated_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame untuk mempercepat proses
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        # Skip frame untuk mengurangi beban
        if frame_id % (skip + 1) == 0:
            results = model(frame, imgsz=320, conf=conf_thres)
            annotated_frame = results[0].plot()

        # Tampilkan frame terakhir yang diproses
        if annotated_frame is not None:
            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        frame_id += 1

        # Delay kecil hanya jika fps rendah
        if delay > 0 and fps < 30:
            time.sleep(delay / 2)

    cap.release()
