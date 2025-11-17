import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import json 
import scipy.stats
import plotly.graph_objects as go
import plotly.express as px
from audiorecorder import audiorecorder 

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Audio Classifier - Buka vs Tutup",
    page_icon="🔊",
    layout="wide"
)

# ===================== CUSTOM CSS (TEMA GELAP) =====================
st.markdown("""
<style>
/* Tema Dark Mode */
body { 
    background-color: #0e1117; /* Latar belakang gelap */
    color: #f8f9fa; /* Teks terang */
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header */
h1, h2, h3, h4 { 
    color: #a5d7ff; /* Biru muda */
    text-align: center; 
    font-weight: 600;
}

/* Tombol */
.stButton > button {
    background: linear-gradient(90deg, #004e92, #000428); /* Gradient biru tua */
    color: white; 
    border-radius: 8px;
    padding: 0.6rem 1.2rem; 
    font-size: 1.05rem;
    font-weight: 500;
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}
.stButton > button:hover {
    background: linear-gradient(90deg, #6a11cb, #2575fc); /* Gradient hover */
    transform: scale(1.03);
}

/* Kartu Hasil */
.result-card {
    background-color: #1a1d23; /* Abu-abu gelap */
    border-radius: 12px;
    padding: 1.5rem; 
    text-align: center;
    margin: 1rem auto; 
    border: 1px solid #343a40; /* Border gelap */
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    max-width: 600px; /* Batasi lebar maksimal */
}

/* Border samping berwarna untuk status */
.result-buka { border-left: 6px solid #28a745; } /* Hijau */
.result-tutup { border-left: 6px solid #dc3545; } /* Merah */
.result-denied { border-left: 6px solid #dc3545; } /* Merah */

/* Teks kecil */
.small-muted { 
    color: #9aa6b2; /* Abu-abu muda */
    font-size: 0.9rem; 
    text-align:center; 
}

/* Ubah warna expander agar terlihat di tema gelap */
.stExpander {
    border: 1px solid #343a40 !important;
    border-radius: 8px !important;
}
.stExpander > summary {
    color: #a5d7ff !important;
}

/* Container tengah untuk hasil */
.center-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ===================== RESOURCE LOADING (DARI PIPELINE KITA) =====================
@st.cache_resource
def load_resources():
    MODEL_PATH = "model_final.pkl"
    SCALER_PATH = "scaler.pkl"
    ENCODER_PATH = "label_encoder.pkl"
    FEATURE_LIST_PATH = "list_fitur_terbaik.json"
    missing = []
    
    try: model = joblib.load(MODEL_PATH)
    except FileNotFoundError: missing.append(MODEL_PATH); model = None
    try: scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError: missing.append(SCALER_PATH); scaler = None
    try: le = joblib.load(ENCODER_PATH)
    except FileNotFoundError: missing.append(ENCODER_PATH); le = None
    try:
        with open(FEATURE_LIST_PATH, 'r') as f: list_fitur_terbaik = json.load(f)
    except FileNotFoundError: missing.append(FEATURE_LIST_PATH); list_fitur_terbaik = None

    if missing:
        raise FileNotFoundError("File(s) tidak ditemukan atau gagal diload:\n- " + "\n- ".join(missing))
    return model, scaler, le, list_fitur_terbaik

# ===================== FEATURE EXTRACTION (DARI SEL 6/14 KITA) =====================
def extract_features_pro(file_path, sr_target=22050, n_mfcc=20):
    """
    Mengekstrak BANYAK FITUR (94 Fitur):
    Persis sama dengan pipeline training kita.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < 1000: return None, None, None
        features = {
            'mean': np.mean(y), 'std': np.std(y), 'skewness': scipy.stats.skew(y),
            'kurtosis': scipy.stats.kurtosis(y), 'iqr': scipy.stats.iqr(y),
            'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
            'rms': np.mean(librosa.feature.rms(y=y))
        }
        S = np.abs(librosa.stft(y))
        features['spec_centroid'] = np.mean(librosa.feature.spectral_centroid(S=S))
        features['spec_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=S))
        features['spec_contrast'] = np.mean(librosa.feature.spectral_contrast(S=S))
        features['spec_flatness'] = np.mean(librosa.feature.spectral_flatness(S=S))
        features['spec_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i+1}_skew'] = scipy.stats.skew(mfccs[i])
            features[f'mfcc_{i+1}_kurt'] = scipy.stats.kurtosis(mfccs[i])
        return features, y, sr
    except Exception as e:
        st.error(f"Gagal saat ekstraksi fitur: {e}")
        return None, None, None

# ===================== PREDICT FUNCTION (DARI SEL 14 KITA) =====================
def predict_audio(file_path, model, scaler, list_fitur_terbaik, le):
    fitur_dict, y, sr = extract_features_pro(file_path)
    if fitur_dict is None:
        return "Error: Gagal memproses file audio.", None, None, None, None, None
    df_single = pd.DataFrame([fitur_dict])
    
    try:
        original_94_features = scaler.feature_names_in_
        df_ordered = df_single[original_94_features]
        X_scaled_all = scaler.transform(df_ordered)
        df_scaled_all = pd.DataFrame(X_scaled_all, columns=original_94_features)
        df_final = df_scaled_all[list_fitur_terbaik] # Pilih fitur terbaik
    except Exception as e:
         return f"Error: Gagal memproses pipeline: {e}", None, None, None, None, None

    prediksi_angka = model.predict(df_final)       
    prediksi_proba_all = model.predict_proba(df_final)[0] 
    prediksi_label = le.inverse_transform(prediksi_angka)[0] 
    confidence = np.max(prediksi_proba_all)
    
    return prediksi_label, confidence, prediksi_proba_all, fitur_dict, y, sr

# ===================== PLOTS (Dengan Tampilan Tema Gelap) =====================
def plot_waveform(y, sr):
    time = np.linspace(0, len(y)/sr, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode="lines", line=dict(color="#2ca8ff")))
    fig.update_layout(title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude",
                        height=250, margin=dict(l=0,r=0,t=40,b=0),
                        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
                        font=dict(color="#f8f9fa"))
    return fig

def plot_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = px.imshow(S_dB, color_continuous_scale="magma", aspect="auto")
    fig.update_layout(title="Mel Spectrogram", xaxis_title="Frame", yaxis_title="Mel bands",
                        height=300, margin=dict(l=0,r=0,t=40,b=0),
                        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
                        font=dict(color="#f8f9fa"))
    return fig

# ===================== MAIN APP =====================
def main():
    st.markdown("<h1>Klasifikasi Suara: Sistem Buka Tutup</h1>", unsafe_allow_html=True)
    
    # --- 1. LOAD RESOURCES ---
    # Muat resources di awal
    try:
        model, scaler, le, list_fitur_terbaik = load_resources()
    except FileNotFoundError as e:
        st.error(f"Gagal memuat resources:\n{str(e)}")
        st.info("Pastikan file 'model_final.pkl', 'scaler.pkl', 'label_encoder.pkl', dan 'list_fitur_terbaik.json' berada di folder yang sama dengan app.py")
        st.stop()
    
    confidence_threshold = 0.80 # Default 80% (Hardcoded)

    # --- 2. BUAT KONTROL INPUT ---
    
    # Variabel untuk menyimpan file yang akan diprediksi
    temp_path_to_predict = None
    run_prediction = False

    col_upload, col_record = st.columns(2)
    
    with col_upload:
        st.markdown("<h4>Upload file untuk Prediksi:</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"], label_visibility="collapsed")

    with col_record:
        st.markdown("<h4>Rekam suara untuk Prediksi:</h4>", unsafe_allow_html=True) # Judul diubah
        audio = audiorecorder("Klik untuk mulai/stop merekam", "Merekam...")

    # --- 3. LOGIKA UNTUK MENANGANI INPUT & TOMBOL ---

    # Logika untuk Kolom Upload
    with col_upload:
        if uploaded_file is not None:
            # Simpan file upload ke temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                temp_path_to_predict = tmp.name # Tetapkan file ini untuk prediksi
            st.audio(uploaded_file, format="audio/wav")

            # Tombol prediksi HANYA muncul jika ada file upload
            if st.button("Jalankan Prediksi (dari File)"):
                run_prediction = True # Set flag untuk prediksi

    # Logika untuk Kolom Rekam
    with col_record:
        if audio is not None and len(audio) > 0:
            st.audio(audio.export(format="wav").read(), format="audio/wav")
            
            # Simpan file rekaman ke temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                
                # Jika user juga upload file, rekaman akan menimpa
                if temp_path_to_predict is not None:
                    try:
                        os.unlink(temp_path_to_predict) # Hapus temp file upload
                    except:
                        pass # Abaikan jika file tidak ada
                
                temp_path_to_predict = tmp.name # Tetapkan file ini untuk prediksi

            # Tombol prediksi HANYA muncul jika ada rekaman
            if st.button("Jalankan Prediksi (dari Rekaman)"):
                run_prediction = True # Set flag untuk prediksi

    # --- 4. TAMPILKAN HASIL PREDIKSI (Blok ini sekarang di luar kolom) ---
    if run_prediction and temp_path_to_predict:
        with st.spinner("Menganalisis audio..."):
            try:
                pred_label, confidence, proba_all, fitur_mentah, y, sr = predict_audio(
                    temp_path_to_predict, model, scaler, list_fitur_terbaik, le
                )
            except Exception as e:
                st.error(f"Kesalahan saat prediksi: {e}")
                if os.path.exists(temp_path_to_predict):
                    os.unlink(temp_path_to_predict) # Hapus temp file jika error
                st.stop() # Hentikan eksekusi

            if proba_all is None: # Cek jika terjadi error
                st.error(pred_label) # 'pred_label' akan berisi pesan error
                if os.path.exists(temp_path_to_predict):
                    os.unlink(temp_path_to_predict) # Hapus temp file jika error
                st.stop() # Hentikan eksekusi

        # --- Logika Verifikasi ---
        akses_diterima = True
        pesan_verifikasi = ""
        
        if pred_label == 'other':
            akses_diterima = False
            pesan_verifikasi = f"AKSES DITOLAK! (Prediksi = 'other')"
        elif confidence < confidence_threshold:
            akses_diterima = False
            pesan_verifikasi = f"AKSES DITOLAK! (Keyakinan Rendah)"
        else:
            akses_diterima = True # Diterima
            pesan_verifikasi = f"AKSES DITERIMA! (Prediksi = {pred_label.upper()})"

        
        # === Tampilkan Hasil ===
        st.markdown("<br><hr>", unsafe_allow_html=True) # Garis pemisah
        
        if akses_diterima:
            st.markdown(f"<h3 style='text-align: center;'>{pesan_verifikasi}</h3>", unsafe_allow_html=True)
            
            # Kartu hasil di tengah
            if "buka" in pred_label.lower():
                card_class = "result-buka"
                card_title = f"PREDIKSI: {pred_label.upper()}"
            elif "tutup" in pred_label.lower():
                card_class = "result-tutup"
                card_title = f"PREDIKSI: {pred_label.upper()}"
            else: 
                card_class = "result-buka" # Default
                card_title = f"PREDIKSI: {pred_label.upper()}"

            st.markdown(f'<div class="result-card {card_class}"><h2>{card_title}</h2>'
                        f'<p style="font-size: 1.2rem; margin:0;">Confidence: <b>{confidence*100:.1f}%</b></p></div>', 
                        unsafe_allow_html=True)

            # Buat 3 kolom untuk konten detail (kiri kosong, tengah isi, kanan kosong)
            col_left, col_center, col_right = st.columns([1, 2, 1])
            
            with col_center:
                # Tampilkan fitur yang KITA gunakan
                st.markdown("<br><b>Fitur yang digunakan (Nilai Mentah):</b>", unsafe_allow_html=True)
                with st.expander("Lihat detail fitur"):
                    for f in list_fitur_terbaik:
                        st.markdown(f"- <code>{f}</code>: <b>{fitur_mentah[f]:.5f}</b>", unsafe_allow_html=True)
                
                # Tampilkan plot probabilitas 5 KELAS KITA
                prob_df = pd.DataFrame({"Kelas": le.classes_, "Probabilitas (%)": proba_all * 100})
                fig_bar = px.bar(prob_df, x="Kelas", y="Probabilitas (%)", color="Kelas",
                                 color_discrete_map={ 
                                     'aqbil_buka': '#28a745', 'reni_buka': '#20c997',
                                     'aqbil_tutup': '#dc3545', 'reni_tutup': '#fd7e14',
                                     'other': '#ffc107'
                                 })
                fig_bar.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#1a1d23",
                                        font=dict(color="#f8f9fa"), height=300)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Plot audio dalam 2 kolom
                col_wave, col_spec = st.columns(2)
                with col_wave:
                    st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
                with col_spec:
                    st.plotly_chart(plot_spectrogram(y, sr), use_container_width=True)
        
        else:
            # --- Jika ditolak, tampilkan format baru ---
            st.markdown(f"<h3 style='text-align: center;'>{pesan_verifikasi}</h3>", unsafe_allow_html=True)
            
            detail_pesan = ""
            if pred_label == 'other':
                detail_pesan = (f"Similarity Score: {confidence:.3f} (Threshold: {confidence_threshold:.2f})\n"
                                f"\nSuara tidak dikenali sebagai pengguna terdaftar\n"
                                f"\nAkses ditolak untuk keamanan sistem")
            elif confidence < confidence_threshold:
                detail_pesan = (f"Similarity Score: {confidence:.3f} (Threshold: {confidence_threshold:.2f})\n"
                                f"\nModel tidak cukup yakin dengan prediksi ({pred_label.upper()})\n"
                                f"\nAkses ditolak untuk keamanan sistem")

            # Tampilkan kartu merah dengan teks di tengah
            st.markdown(f'<div class="result-card result-denied">'
                        f'<h2>AKSES DITOLAK</h2>'
                        f'<pre style="text-align: center; font-size: 1.1rem; color: #f8f9fa; background: #1a1d23; border: none; padding: 0; white-space: pre-wrap; font-family: inherit;">{detail_pesan}</pre></div>', 
                        unsafe_allow_html=True)
            
            col_left, col_center, col_right = st.columns([1, 2, 1])
            
            with col_center:
                # Tampilkan juga rincian probabilitas (ini berguna untuk debug)
                st.subheader("Detail Probabilitas")
                prob_df = pd.DataFrame({"Kelas": le.classes_, "Probabilitas (%)": proba_all * 100})
                fig_bar = px.bar(prob_df, x="Kelas", y="Probabilitas (%)", color="Kelas",
                                 color_discrete_map={
                                     'aqbil_buka': '#28a745', 'reni_buka': '#20c997',
                                     'aqbil_tutup': '#dc3545', 'reni_tutup': '#fd7e14',
                                     'other': '#ffc107'
                                 })
                fig_bar.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#1a1d23",
                                        font=dict(color="#f8f9fa"), height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
            
        # Hapus file temp di luar blok 'if', agar selalu dijalankan
        if os.path.exists(temp_path_to_predict):
            os.unlink(temp_path_to_predict)

if __name__ == "__main__":
    main()