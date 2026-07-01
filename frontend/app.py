import streamlit as st
import requests
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MNIST Neural Classifier",
    page_icon="◈",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Animated grid background, evokes a network "reading" pixels ---- */
.stApp {
    background-color: #0B0E1A;
    background-image:
        linear-gradient(rgba(99, 102, 241, 0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99, 102, 241, 0.07) 1px, transparent 1px);
    background-size: 28px 28px;
    background-position: center;
}

/* ---- Hero header ---- */
.hero-wrap {
    text-align: center;
    padding: 18px 0 8px 0;
}

.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    color: #6366F1;
    font-size: 13px;
    letter-spacing: 3px;
    font-weight: 600;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 700;
    color: #F1F5F9;
    margin: 6px 0 4px 0;
    letter-spacing: -0.5px;
}

.hero-title span {
    color: #6366F1;
}

.hero-subtitle {
    color: #64748B;
    font-size: 15px;
    font-weight: 400;
    margin-bottom: 28px;
}

/* ---- Upload zone container ---- */
.upload-card {
    background: #12162A;
    border: 1px solid #1E2438;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

div[data-testid="stFileUploader"] {
    border-radius: 12px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #161B2E;
    border: 1.5px dashed #2D3454;
    border-radius: 12px;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6366F1;
}

/* ---- Preview panel ---- */
.preview-label {
    font-family: 'Space Grotesk', sans-serif;
    color: #94A3B8;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.scan-frame {
    border: 1px solid #2D3454;
    border-radius: 14px;
    padding: 14px;
    background: #0F1322;
    text-align: center;
    position: relative;
}

.scan-frame img {
    border-radius: 8px;
}

/* ---- Predict button ---- */
div.stButton > button {
    background: linear-gradient(135deg, #6366F1, #4F46E5);
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 10px;
    padding: 12px 0;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35);
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.55);
    transform: translateY(-1px);
}

div.stButton > button:active {
    transform: translateY(0px);
}

/* ---- Result card ---- */
.result-box {
    margin-top: 8px;
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(160deg, #161B2E 0%, #12162A 100%);
    border: 1px solid #2D3454;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-box::before {
    content: "";
    position: absolute;
    top: -40%;
    left: 50%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.18), transparent 70%);
    transform: translateX(-50%);
    pointer-events: none;
}

.result-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #64748B;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 14px;
    position: relative;
    z-index: 1;
}

.result-digit {
    font-family: 'Space Grotesk', sans-serif;
    color: #F59E0B;
    font-size: 76px;
    font-weight: 700;
    line-height: 1;
    text-shadow: 0 0 30px rgba(245, 158, 11, 0.4);
    position: relative;
    z-index: 1;
}

.result-caption {
    color: #475569;
    font-size: 13px;
    margin-top: 10px;
    position: relative;
    z-index: 1;
}

/* ---- Info hint box ---- */
.hint-box {
    background: #161B2E;
    border-left: 3px solid #6366F1;
    border-radius: 8px;
    padding: 14px 16px;
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 16px;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    color: #2D3454;
    margin-top: 60px;
    font-size: 12px;
    letter-spacing: 1px;
    font-family: 'Space Grotesk', sans-serif;
}

.footer span {
    color: #475569;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown(
    """
    <div class='hero-wrap'>
        <div class='hero-eyebrow'>Deep Learning · Computer Vision</div>
        <div class='hero-title'>MNIST <span>Neural</span> Classifier</div>
        <div class='hero-subtitle'>Upload a handwritten digit — the network reads it pixel by pixel</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------

st.markdown("<div class='upload-card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a digit image (0–9)",
    type=["png", "jpg", "jpeg"],
    label_visibility="visible"
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- IMAGE PREVIEW + PREDICT ----------------

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='preview-label'>Input</div>", unsafe_allow_html=True)
        st.markdown("<div class='scan-frame'>", unsafe_allow_html=True)
        st.image(image, width=180)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='preview-label'>Action</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hint-box'>Network expects a single digit, "
            "centered, on a plain background.</div>",
            unsafe_allow_html=True
        )
        predict_btn = st.button("Run Prediction", use_container_width=True)

    # ---------------- PREDICTION ----------------

    if predict_btn:

        try:
            with st.spinner("Reading pixels through the network..."):

                files = {
                    "file": uploaded_file.getvalue()
                }

                response = requests.post(
                    "https://farooqdaniyal-wahid-miniest-backend.hf.space/predict",
                    ## "https://farooqdaniyal-mnist-backend.hf.space/predict",
                    files=files,
                    timeout=20
                )

                prediction = response.json()["prediction"]

            st.markdown(
                f"""
                <div class='result-box'>
                    <div class='result-title'>Predicted Digit</div>
                    <div class='result-digit'>{prediction}</div>
                    <div class='result-caption'>Inference complete</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Backend connection failed — {e}")

# ---------------- FOOTER ----------------

st.markdown(
    """
    <div class='footer'>
        BUILT WITH <span>TENSORFLOW · FASTAPI · STREAMLIT</span>
    </div>
    """,
    unsafe_allow_html=True
)