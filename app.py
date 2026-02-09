import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import sys
import plotly.express as px
import pandas as pd
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classifier import GastroClassifier
from xai.grad_cam import GradCAM, overlay_cam
from xai.similarity import SimilaritySearch
from xai.counterfactual import CounterfactualGenerator
from data.loader import KvasirDataset
from torch.utils.data import DataLoader

# Constants
CLASSES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
MODEL_PATH = "models/best_model.pth"
DATA_DIR = "data/kvasir-dataset-v2"

# Page Config
st.set_page_config(
    page_title="Gastro XAI Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

# Custom CSS (Matched to Screenshot Version)
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6; 
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/gastroenterology.png", width=80)
    st.title("Gastro XAI")
    st.markdown("---")
    st.subheader("Analysis Settings")
    enable_gradcam = st.toggle("Visual Attention (Grad-CAM)", value=True)
    enable_similarity = st.toggle("Similar Historical Cases", value=True)
    enable_counterfactual = st.toggle("Counterfactual Analysis", value=False)
    
    st.markdown("---")
    st.success("System: **Online**") 
    st.info(f"Model: ResNet50\nClasses: {len(CLASSES)}")

# Load Model
@st.cache_resource
def load_model():
    model = GastroClassifier(num_classes=len(CLASSES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please run training first.")
        return None
    model.eval()
    return model

model = load_model()

# Load Similarity Search (Lazy Loading preserved)
@st.cache_resource
def load_similarity_engine():
    if not os.path.exists(DATA_DIR): return None
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        dataset = KvasirDataset(DATA_DIR, subset='train', transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        return SimilaritySearch(model, loader, device='cpu')
    except Exception as e:
        return None

sim_search = None
if enable_similarity and model:
    # Check if we should load it (lazy load)
    if 'sim_engine_loaded' not in st.session_state:
        st.sidebar.warning("⚠️ Knowledge Base Paused")
        if st.sidebar.button("Initialize Knowledge Base"):
            with st.spinner("Indexing Clinical Cases..."):
                sim_search = load_similarity_engine()
                st.session_state['sim_engine_loaded'] = True
                st.rerun()
    else:
        sim_search = load_similarity_engine()

# Header
st.title("🩺 Gastrointestinal Disease AI Diagnostic Assistant")
st.markdown("#### Upload an endoscopy image for real-time classification and explainable insights.")

# Main Interface
uploaded_file = st.file_uploader("Drop your scan here", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, PNG")

if uploaded_file is not None and model:
    image = Image.open(uploaded_file).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        pred_class = CLASSES[pred_idx.item()]
    inference_time = (time.time() - start_time) * 1000

    col_img, col_metrics = st.columns([1, 2])
    
    with col_img:
        st.image(image, caption='Query Image', width="stretch")
    
    with col_metrics:
        # Exact formatting from screenshot: Diagnosis: [Class] (Blue)
        st.markdown(f"### Diagnosis: :blue[{pred_class.replace('-', ' ').title()}]")
        
        # Metrics: Confidence Score, Model Uncertainty, Inference Time
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Confidence Score", f"{confidence.item():.1%}")
        with m2:
            st.metric("Model Uncertainty", f"{(1 - confidence.item()):.1%}")
        with m3:
            st.metric("Inference Time", f"~{int(inference_time)}ms")
            
        # Probability Chart: Blues scale
        probs_np = probs[0].numpy()
        df_probs = pd.DataFrame({'Class': CLASSES, 'Probability': probs_np})
        df_probs = df_probs.sort_values('Probability', ascending=True)
        
        fig = px.bar(
            df_probs, 
            x='Probability', 
            y='Class', 
            orientation='h',
            text_auto='.1%',
            title="Class Probability Distribution",
            color='Probability',
            color_continuous_scale='Blues' # Matches screenshot
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧩 Explainable Insights")
    tab1, tab2, tab3 = st.tabs(["🔍 Visual Attention", "📚 Similar Cases", "🔄 Counterfactuals"])
    
    with tab1:
        if enable_gradcam:
            col_cam1, col_cam2 = st.columns([1, 2])
            with col_cam1:
                try:
                    target_layer = model.backbone.layer4[2]
                    grad_cam = GradCAM(model, target_layer)
                    cam, _ = grad_cam(input_tensor, pred_idx)
                    overlay = overlay_cam(input_tensor[0], cam)
                    
                    st.image(overlay, caption=f"Grad-CAM Attention Map", width="stretch")
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
            with col_cam2:
                # Info box matching screenshot
                st.info("""
                **What am I looking at?**
                - The **red/yellow** regions indicate the specific parts of the image the model focused on to make its prediction.
                - **Interpretation**: If the model predicts "Polyps" and the red area covers the polyp, the model is looking at the correct feature.
                """)
        else:
            st.warning("Grad-CAM analysis is disabled in sidebar.")

    with tab2:
        if enable_similarity and sim_search:
            st.markdown(f"**Historical cases similar to this patient:**")
            try:
                distances, indices = sim_search.search(input_tensor)
                
                cols = st.columns(4)
                for i, idx in enumerate(indices[0][:4]):
                    path = sim_search.image_paths[idx]
                    label = sim_search.labels[idx]
                    class_name = CLASSES[label]
                    dist = distances[0][i]
                    
                    with cols[i]:
                        st.image(path, width="stretch")
                        st.caption(f"**{class_name}**\nS: {1/(1+dist):.2f}")
            except Exception as e:
                st.error(f"Search failed: {e}")
        else:
            if enable_similarity:
                st.info("Knowledge Base not initialized. Check sidebar.")
            else:
                st.warning("Similarity Search disabled.")

    with tab3:
        if enable_counterfactual:
            col_cf1, col_cf2 = st.columns([1, 1])
            with col_cf1:
                 st.markdown("Generates a minimal perturbation to change the model's prediction.")
                 try:
                    target_idx = (pred_idx + 1) % len(CLASSES)
                    target_class = CLASSES[target_idx]
                    
                    cf_gen = CounterfactualGenerator(model)
                    with st.spinner(f"Generating counterfactual to flip to {target_class}..."):
                        perturbed_img = cf_gen.generate(input_tensor, target_class=target_idx, steps=30)
                    
                    if perturbed_img is not None:
                        # Denormalize
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        disp_img = perturbed_img[0].cpu() * std + mean
                        disp_img = torch.clamp(disp_img, 0, 1)
                        disp_img = transforms.ToPILImage()(disp_img)
                        st.image(disp_img, caption=f"Counterfactual Example ({target_class})", width=300)
                    else:
                        st.warning("Could not generate a counterfactual within the step limit.")
                 except Exception as e:
                     st.error(f"Counterfactual generation error: {e}")
            with col_cf2:
                st.info("Counterfactuals help understand decision boundaries. The image on the left is the 'closest' visual modification required to change the diagnosis.")
        else:
            st.warning("Counterfactual analysis is disabled (Computationally Expensive). Enable in sidebar.")
