import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import tempfile
import plotly.graph_objects as go
import plotly.express as px

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Audio Classifier - Buka vs Tutup",
    page_icon="",
    layout="wide"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
body { background-color: #0e1117; color: #f8f9fa; }
h1, h2, h3, h4 { color: #a5d7ff; text-align: center; }
.stButton > button {
    background: linear-gradient(90deg, #004e92, #000428);
    color: white; border-radius: 8px;
    padding: 0.6rem 1.2rem; font-size: 1.05rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    transform: scale(1.05);
}
.result-card {
    background-color: #1a1d23; border-radius: 12px;
    padding: 1rem; text-align: center;
    margin-top: 1rem; box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
}
.result-buka { border: 2px solid #3df55e; }
.result-tutup { border: 2px solid #ff4d4d; }
.footer {
    text-align: center; color: #8c8c8c;
    margin-top: 3rem; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL / SCALER / ENCODER =====================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le, True
    except Exception as e:
        st.error(f"Gagal memuat model atau file pendukung: {e}")
        return None, None, None, False

# ===================== FEATURE EXTRACTION =====================
def extract_features(y, sr=22050):
    y = y / (np.max(np.abs(y)) + 1e-6)
    feats = [
        np.mean(y), np.std(y), np.var(y),
        np.sqrt(np.mean(y**2)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.min(y), np.max(y), np.ptp(y),
        np.median(y), stats.iqr(y),
        np.sum(y**2),
        -np.sum((y**2) * np.log(y**2 + 1e-12)),
        stats.skew(y), stats.kurtosis(y),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_flatness(y=y)),
        np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        -np.sum(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) * np.log(np.abs(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) + 1e-12))
    ]
    return pd.DataFrame([feats])

# ===================== AUDIO PREPROCESSING =====================
def preprocess_audio(y, sr, target_sr=22050, duration=1.0):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    max_len = int(sr * duration)
    y = np.pad(y, (0, max_len - len(y)), mode="constant") if len(y) < max_len else y[:max_len]
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y, sr

# ===================== PREDICT FUNCTION =====================
def predict_audio(file, model, scaler, le):
    y, sr = librosa.load(file, sr=None)
    y, sr = preprocess_audio(y, sr)
    X_new = extract_features(y, sr)

    # Normalisasi fitur
    X_scaled = scaler.transform(X_new)

    pred_enc = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    pred_label = le.inverse_transform([pred_enc])[0]
    return pred_label, proba, y, sr

# ===================== PLOTS =====================
def plot_waveform(y, sr):
    time = np.linspace(0, len(y)/sr, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode="lines", line=dict(color="#2ca8ff")))
    fig.update_layout(
        title="Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#f8f9fa")
    )
    return fig

def plot_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = px.imshow(S_dB, color_continuous_scale="magma", aspect="auto")
    fig.update_layout(
        title="Mel Spectrogram",
        xaxis_title="Frame", yaxis_title="Mel bands",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#f8f9fa")
    )
    return fig

# ===================== MAIN APP =====================
def main():
    st.markdown("<h1>Audio Classifier: Suara Buka vs Tutup</h1>", unsafe_allow_html=True)
    st.write("Upload audio berdurasi **1 detik (WAV)** untuk mendeteksi apakah suara tersebut adalah **‘Buka’** atau **‘Tutup’.**")

    model, scaler, le, loaded = load_resources()
    if not loaded:
        st.stop()

    uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.audio(uploaded_file, format="audio/wav")

        if st.button("Jalankan Prediksi"):
            with st.spinner("Menganalisis audio..."):
                pred_label, proba, y, sr = predict_audio(temp_path, model, scaler, le)

            col1, col2 = st.columns([1, 1])

            with col1:
                if pred_label.lower() == "buka":
                    st.markdown(f'<div class="result-card result-buka"><h2>PREDIKSI: BUKA</h2><p>Confidence: {max(proba)*100:.1f}%</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-card result-tutup"><h2>PREDIKSI: TUTUP</h2><p>Confidence: {max(proba)*100:.1f}%</p></div>', unsafe_allow_html=True)

                prob_df = pd.DataFrame({"Kelas": le.classes_, "Probabilitas (%)": proba * 100})
                fig_bar = px.bar(prob_df, x="Kelas", y="Probabilitas (%)",
                                 color="Kelas", color_discrete_sequence=["#3df55e", "#ff4d4d"])
                fig_bar.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                      font=dict(color="#f8f9fa"), height=300)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
                st.plotly_chart(plot_spectrogram(y, sr), use_container_width=True)

        os.unlink(temp_path)

# ===================== RUN =====================
if __name__ == "__main__":
    main()
