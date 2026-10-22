"""
Streamlit Web Application for Plant Disease Detection
Powered by DAA-MDeNet trained model.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

# Ensure models module can be imported
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.daa_mdenet import DAAMDeNet

st.set_page_config(
    page_title="Plant Disease Detector | DAA-MDeNet",
    page_icon="🌿",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4e555b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f1f8e9;
        border-left: 6px solid #43a047;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by <b>DAA-MDeNet</b> (Dilated Adaptive Attention Multiscale DenseNet)</div>', unsafe_allow_html=True)

# Candidate paths for model
CANDIDATE_PATHS = [
    PROJECT_ROOT / "daa_mdenet_best.pth",
    Path(r"C:\Users\USER\Downloads\daa_mdenet_best.pth"),
]

@st.cache_resource
def load_trained_model():
    model_path = None
    for p in CANDIDATE_PATHS:
        if p.exists():
            model_path = p
            break
            
    if model_path is None:
        return None, None, None

    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint.get("classes", [])
    num_classes = len(classes) if classes else 37

    model = DAAMDeNet(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_acc = checkpoint.get("val_acc", None)
    return model, classes, val_acc

model, class_names, best_acc = load_trained_model()

# Sidebar Info
with st.sidebar:
    st.header("⚙️ Model Status")
    if model is not None:
        st.success("✅ Model loaded successfully!")
        if best_acc is not None:
            st.metric(label="Validation Accuracy", value=f"{best_acc:.2f}%")
        st.write(f"**Recognized Classes:** {len(class_names)}")
    else:
        st.error("❌ Model checkpoint `daa_mdenet_best.pth` not found in Downloads or Project directory.")
    
    st.markdown("---")
    st.subheader("📖 About")
    st.info(
        "This system uses **DAA-MDeNet** with multiscale dilated dense blocks and a self-attention mechanism "
        "to classify plant leaf diseases across 37 categories."
    )

# Preprocessing transforms (matches training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose an image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
    else:
        st.write("👈 Upload an image to analyze, or use one from your dataset!")

with col2:
    st.subheader("🔍 Disease Diagnosis")
    if uploaded_file is not None and model is not None:
        with st.spinner("Analyzing leaf with DAA-MDeNet..."):
            img_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                top_prob, top_idx = torch.topk(probs, 5)

            predicted_class = class_names[top_idx[0].item()]
            confidence = top_prob[0].item() * 100

            is_healthy = "healthy" in predicted_class.lower()

            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin-top:0; color:{'#2e7d32' if is_healthy else '#c62828'};">
                    {'🌱 Healthy Leaf' if is_healthy else '⚠️ Disease Detected'}
                </h3>
                <h2 style="margin: 0.3rem 0; color: #212529;">{predicted_class}</h2>
                <p style="font-size: 1.1rem; color: #495057;">Confidence: <b>{confidence:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Top 5 Predictions")
            for i in range(len(top_idx)):
                cls_name = class_names[top_idx[i].item()]
                prob = top_prob[i].item() * 100
                st.write(f"**{cls_name}** ({prob:.1f}%)")
                st.progress(prob / 100.0)

    elif uploaded_file is not None and model is None:
        st.warning("Please ensure `daa_mdenet_best.pth` is present to run inference.")
    else:
        st.info("Results will appear here once an image is uploaded.")
