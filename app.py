import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Book Detector",
    page_icon="📚",
    layout="wide"
)

# --- SIDEBAR PENGATURAN ---
st.sidebar.title("⚙️ Pengaturan")

st.sidebar.markdown("Atur sensitivitas deteksi model di sini:")

# Slider Confidence Threshold
conf_thresh = st.sidebar.slider(
    "Confidence threshold", 
    min_value=0.0, max_value=1.0, value=0.40, step=0.01,
    help="Semakin tinggi, model semakin ketat/yakin sebelum mendeteksi objek sebagai buku."
)

# Slider IoU Threshold (NMS)
iou_thresh = st.sidebar.slider(
    "IoU threshold (NMS)", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.01,
    help="Mengatur seberapa ketat model menghapus kotak (bounding box) yang tumpang tindih pada buku yang sama."
)

# --- LOAD MODEL (Di-cache agar cepat) ---
@st.cache_resource
def load_model():
    # Mengambil model terbaik dari folder train-2 hasil eksperimen terakhir
    return YOLO('runs/detect/train-2/weights/best.pt')

model = load_model()

# --- MAIN CONTENT ---
st.title("📚 Book Detector")
st.write("Upload foto rak, meja, atau tumpukan barang untuk mendeteksi buku secara otomatis!")
st.markdown("---")

st.write("**Upload foto (JPG/PNG/WEBP)**")
uploaded_photo = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], key="photo")
    
if uploaded_photo:
    image = Image.open(uploaded_photo)
    
    # Buat 2 kolom untuk perbandingan
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Hasil Deteksi")
        with st.spinner("Mendeteksi buku..."):
            # Menjalankan prediksi dengan parameter dari sidebar
            # device=0 memastikan prediksi berjalan secepat kilat menggunakan GPU Nvidia
            results = model.predict(
                source=image,
                conf=conf_thresh,
                iou=iou_thresh,
                device=0 
            )
            
            # Menggambar hasil bounding box ke atas gambar
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_rgb, use_container_width=True)
            
    # Menampilkan ringkasan deteksi di bagian bawah
    jumlah_buku = len(results[0].boxes)
    if jumlah_buku > 0:
        st.success(f"🎉 Berhasil mendeteksi **{jumlah_buku}** buku dalam gambar!")
    else:
        st.warning("Belum ada buku yang terdeteksi. Coba turunkan *Confidence threshold* di panel kiri.")

st.markdown("---")
st.caption("Model: YOLOv8 custom training (Single Class: Book). GPU Accelerated (CUDA).")