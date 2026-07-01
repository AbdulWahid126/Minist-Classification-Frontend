import streamlit as st
import requests
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MNIST Neural Classifier",
    page_icon="▓",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"], p, span, div, label, input, textarea {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ---- Base canvas: pure black with a faint CRT scanline sweep ---- */
.stApp {
    background-color: #000000;
    background-image:
        linear-gradient(rgba(0, 255, 65, 0.035) 1px, transparent 1px);
    background-size: 100% 3px;
}

.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.6) 100%);
    z-index: 0;
}

.block-container {
    max-width: 740px;
    padding-top: 2.2rem;
}

/* ---- Terminal window chrome around the whole page ---- */
.term-titlebar {
    background: #0A0F0A;
    border: 1px solid #1F3D1F;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    padding: 9px 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.term-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    background: #1F3D1F;
}

.term-titlebar-label {
    margin-left: 8px;
    color: #3D8B3D;
    font-size: 12px;
    letter-spacing: 1px;
}

.term-body {
    background: #050805;
    border: 1px solid #1F3D1F;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 26px 28px 20px 28px;
}

/* ---- Hero header ---- */
.hero-wrap {
    padding: 4px 0 26px 0;
}

.hero-eyebrow {
    color: #3D8B3D;
    font-size: 12px;
    letter-spacing: 2px;
}

.hero-eyebrow::before {
    content: "# ";
    color: #1F3D1F;
}

.hero-title {
    font-size: 34px;
    font-weight: 700;
    color: #00FF41;
    margin: 10px 0 10px 0;
    letter-spacing: -0.5px;
    text-shadow: 0 0 14px rgba(0, 255, 65, 0.45);
}

.hero-title span {
    color: #66FF99;
}

.hero-subtitle {
    color: #2E7D2E;
    font-size: 14px;
    margin-bottom: 2px;
}

.hero-subtitle .prompt {
    color: #00FF41;
}

.blink-cursor {
    display: inline-block;
    width: 8px;
    height: 14px;
    background: #00FF41;
    margin-left: 4px;
    animation: blink 1.1s steps(1) infinite;
    vertical-align: -2px;
}

@keyframes blink {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
}

/* ---- Upload zone container ---- */
.upload-card {
    background: #050805;
    border: 1px dashed #1F3D1F;
    border-radius: 4px;
    padding: 20px 22px;
    margin: 20px 0 18px 0;
    position: relative;
}

.upload-card::before {
    content: "[ input_stream ]";
    position: absolute;
    top: -9px;
    left: 16px;
    background: #050805;
    padding: 0 8px;
    font-size: 10px;
    letter-spacing: 1px;
    color: #2E7D2E;
}

div[data-testid="stFileUploader"] {
    border-radius: 4px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #0A0F0A;
    border: 1px solid #1F3D1F !important;
    border-radius: 4px;
    transition: border-color 0.2s ease, background 0.2s ease;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00FF41 !important;
    background: #0D140D;
}

div[data-testid="stFileUploaderDropzone"] * {
    color: #3D8B3D !important;
}

div[data-testid="stFileUploader"] button {
    background: #0A0F0A !important;
    color: #00FF41 !important;
    border: 1px solid #1F3D1F !important;
    border-radius: 3px !important;
}

/* ---- Preview panel ---- */
.preview-label {
    color: #2E7D2E;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.preview-label::before {
    content: "> ";
    color: #00FF41;
}

.scan-frame {
    border: 1px solid #1F3D1F;
    border-radius: 4px;
    padding: 14px;
    background: #030503;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.scan-frame::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00FF41, transparent);
    animation: scanline 2.6s linear infinite;
    opacity: 0.7;
}

@keyframes scanline {
    0% { top: 0%; }
    100% { top: 100%; }
}

.scan-frame img {
    border-radius: 2px;
    position: relative;
    z-index: 1;
    filter: grayscale(15%);
}

/* ---- Action hint + button ---- */
.hint-box {
    background: #0A0F0A;
    border-left: 2px solid #00FF41;
    border-radius: 2px;
    padding: 12px 14px;
    color: #4CAF50;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 16px;
}

.hint-box::before {
    content: "// ";
    color: #1F3D1F;
}

div.stButton > button {
    background: #0A0F0A;
    color: #00FF41;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.5px;
    border: 1px solid #00FF41;
    border-radius: 3px;
    padding: 11px 0;
    box-shadow: 0 0 0px rgba(0, 255, 65, 0);
    transition: all 0.15s ease;
}

div.stButton > button:hover {
    background: #00FF41;
    color: #000000;
    box-shadow: 0 0 22px rgba(0, 255, 65, 0.55);
}

div.stButton > button:active {
    transform: translateY(1px);
}

div.stButton > button:focus-visible {
    outline: 1px solid #66FF99;
    outline-offset: 2px;
}

div.stButton > button p {
    color: inherit !important;
}

/* ---- Result card ---- */
.result-box {
    margin-top: 6px;
    padding: 28px 26px;
    border-radius: 4px;
    background: #030503;
    border: 1px solid #1F3D1F;
    text-align: left;
    position: relative;
}

.result-title {
    color: #2E7D2E;
    font-size: 11px;
    letter-spacing: 2px;
    margin-bottom: 6px;
}

.result-title::before {
    content: "$ ";
    color: #00FF41;
}

.result-digit-row {
    display: flex;
    align-items: baseline;
    gap: 18px;
    margin: 10px 0 14px 2px;
}

.result-digit {
    color: #00FF41;
    font-size: 72px;
    font-weight: 700;
    line-height: 1;
    text-shadow: 0 0 24px rgba(0, 255, 65, 0.55);
}

.result-caption {
    color: #2E7D2E;
    font-size: 12px;
    letter-spacing: 0.5px;
}

.result-caption::before {
    content: "[ OK ] ";
    color: #00FF41;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    color: #1A2E1A;
    margin-top: 40px;
    font-size: 11px;
    letter-spacing: 1px;
}

.footer span {
    color: #2E5A2E;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Error box recolor to fit theme */
div[data-testid="stAlert"] {
    background: #0A0F0A !important;
    border: 1px solid #7A1F1F !important;
    color: #FF6B6B !important;
    border-radius: 4px !important;
}

/* Spinner text recolor */
div[data-testid="stSpinner"] > div {
    color: #00FF41 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TERMINAL SHELL START ----------------

st.markdown(
    """
    <div class='term-titlebar'>
        <span class='term-dot' style='background:#7A2E2E;'></span>
        <span class='term-dot' style='background:#7A6E2E;'></span>
        <span class='term-dot' style='background:#2E7A3D;'></span>
        <span class='term-titlebar-label'>user@classifier: ~/mnist</span>
    </div>
    <div class='term-body'>
    """,
    unsafe_allow_html=True
)

# ---------------- HERO ----------------

st.markdown(
    """
    <div class='hero-wrap'>
        <div class='hero-eyebrow'>deep_learning / computer_vision</div>
        <div class='hero-title'>MNIST <span>Neural</span> Classifier</div>
        <div class='hero-subtitle'><span class='prompt'>$</span> upload a handwritten digit — the network reads it pixel by pixel<span class='blink-cursor'></span></div>
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
        st.markdown("<div class='preview-label'>input</div>", unsafe_allow_html=True)
        st.markdown("<div class='scan-frame'>", unsafe_allow_html=True)
        st.image(image, width=180)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='preview-label'>action</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hint-box'>Network expects a single digit, "
            "centered, on a plain background.</div>",
            unsafe_allow_html=True
        )
        predict_btn = st.button("./run_prediction.sh", use_container_width=True)

    # ---------------- PREDICTION ----------------

    if predict_btn:

        try:
            with st.spinner("reading pixels through the network..."):

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
                    <div class='result-title'>predict --input=upload</div>
                    <div class='result-digit-row'>
                        <div class='result-digit'>{prediction}</div>
                    </div>
                    <div class='result-caption'>inference complete</div>
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
        built with <span>tensorflow · fastapi · streamlit</span>
    </div>
    </div>
    """,
    unsafe_allow_html=True
)