import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import plotly.express as px
import time

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Green Coffee Bean Classifier | Deep Learning",
    page_icon="🌱",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
    .main-title {font-size: 42px; font-weight: 800; color: #4A2E1B; text-align: center;}
    .sub-title {font-size: 18px; color: #7F8C8D; text-align: center; margin-bottom: 30px;}
    .result-card {background-color: #F8F9FA; padding: 25px; border-radius: 15px; 
                  box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center; 
                  border-left: 5px solid #6F4E37;}
    .confidence-text {font-size: 24px; font-weight: bold; color: #2E86C1;}
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_cnn_model():
    return load_model("model/coffee_model.h5")

try:
    model = load_cnn_model()
except:
    st.error("⚠️ Model tidak ditemukan! Pastikan folder model benar.")
    st.stop()

# =========================
# KELAS (HARUS SESUAI TRAIN)
# =========================
classes = ['arabika', 'excelsa', 'liberika', 'robusta']

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=100)
    st.markdown("## 📖 Tentang Aplikasi")
    st.info("Model CNN MobileNetV2 untuk klasifikasi biji kopi hijau (Green Beans).")

    st.markdown("### 🌱 Jenis Kopi Hijau:")
    with st.expander("Arabika"):
        st.write("Bentuk pipih memanjang, garis tengah bergelombang, umumnya berwarna hijau kebiruan.")
    with st.expander("Robusta"):
        st.write("Bentuk lebih bulat dan kecil, garis tengah lurus, warna cenderung hijau kekuningan pucat.")
    with st.expander("Liberika"):
        st.write("Biji berukuran paling besar, bentuk asimetris memanjang.")
    with st.expander("Excelsa"):
        st.write("Bentuk asimetris mirip Liberika namun ukuran lebih kecil.")

# =========================
# HEADER
# =========================
st.markdown('<p class="main-title">🌱 Green Coffee Bean Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Deteksi jenis biji kopi mentah menggunakan CNN</p>', unsafe_allow_html=True)

st.markdown("---")

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload gambar biji kopi hijau", type=["jpg","png","jpeg"])

# =========================
# PREDIKSI
# =========================
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img, use_container_width=True, caption="Gambar Input")

    with col2:
        with st.spinner('AI sedang menganalisis...'):
            time.sleep(1)

            # FIX UTAMA: 256px
            img_resized = img.resize((256, 256))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0]
            idx = np.argmax(pred)
            confidence = pred[idx] * 100

        # RESULT CARD
        st.markdown(f"""
            <div class="result-card">
                <h3 style="color:#6F4E37;">Prediksi: {classes[idx]}</h3>
                <p class="confidence-text">{confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # DATAFRAME
        df = pd.DataFrame({
            'Jenis Kopi': classes,
            'Probabilitas': np.round(pred * 100, 2)
        })

        # GRAFIK
        fig = px.bar(
            df,
            x='Probabilitas',
            y='Jenis Kopi',
            orientation='h',
            color='Probabilitas',
            color_continuous_scale='YlOrBr',
            text='Probabilitas'
        )

        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="",
            yaxis={'categoryorder':'total ascending'},
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False
        )

        fig.update_traces(texttemplate='%{text}%', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload gambar untuk mulai klasifikasi")