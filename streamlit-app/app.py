import os
import json
import base64
from datetime import datetime
from PIL import Image
import streamlit as st
from model_helper import predict_detailed, class_names, CLASS_METADATA, device

# ---------------------------------------------------------
# Icon Management (Flaticon Icons Integration)
# ---------------------------------------------------------
ICON_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'icons')

def get_icon_b64(filename):
    """Returns base64-encoded string for local Flaticon icons directly without stale cache."""
    path = os.path.join(ICON_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def flaticon(name, size=20, extra_style=""):
    """Renders a Flaticon PNG icon as an inline base64 HTML image."""
    fname = f"{name}.png" if not name.endswith('.png') else name
    b64 = get_icon_b64(fname)
    if b64:
        return f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" style="vertical-align: middle; display: inline-block; object-fit: contain; {extra_style}" alt="{name}"/>'
    return ""

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
car_icon_path = os.path.join(ICON_DIR, 'car.png')
page_icon = Image.open(car_icon_path) if os.path.exists(car_icon_path) else "🚗"

st.set_page_config(
    page_title="AutoDamage AI | Vehicle Damage Detection",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling (with Flaticon button icons)
# ---------------------------------------------------------
b64_trash = get_icon_b64('trash.png')
b64_reset = get_icon_b64('reset.png')

st.markdown("""
<style>
    /* Global Font & Spacing Enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Hero Section */
    .hero-container {
        padding: 1.8rem 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
        max-width: 800px;
        line-height: 1.5;
    }
    
    /* Sleek Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    
    /* Severity Status Banner */
    .status-banner {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .status-banner.severe {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .status-banner.moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.25) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .status-banner.normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    /* Probability Bar Container */
    .prob-bar-container {
        margin-bottom: 0.75rem;
    }
    .prob-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 4px;
        color: #cbd5e1;
    }
    .prob-track {
        width: 100%;
        height: 9px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Sidebar aesthetic */
    section[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    /* Streamlit button style refinements */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Button Flaticon Icon CSS (safely rendered with escaped braces)
if b64_reset or b64_trash:
    st.markdown(f"""
    <style>
        /* Reset Button Flaticon Icon */
        div.st-key-btn_reset_app button {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
        }}
        div.st-key-btn_reset_app button::before {{
            content: "" !important;
            display: inline-block !important;
            width: 17px !important;
            height: 17px !important;
            background-image: url('data:image/png;base64,{b64_reset}') !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            flex-shrink: 0 !important;
        }}

        /* Clear Image Button Flaticon Icon */
        div.st-key-btn_clear_image button {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
        }}
        div.st-key-btn_clear_image button::before {{
            content: "" !important;
            display: inline-block !important;
            width: 17px !important;
            height: 17px !important;
            background-image: url('data:image/png;base64,{b64_trash}') !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            flex-shrink: 0 !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
        {flaticon('car', 30)}
        <h2 style="margin:0; font-size:1.4rem; font-weight:800; color:#f8fafc;">AutoDamage AI</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Deep Learning Vehicle Triage Platform")
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        {flaticon('brain', 18)}
        <h4 style="margin:0; font-size:1rem; font-weight:700;">Model Specs</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    - **Architecture**: ResNet-50 (Fine-Tuned)
    - **Backbone**: PyTorch Deep CNN
    - **Input Resolution**: 224 × 224 px
    - **Inference Device**: `{str(device).upper()}`
    - **Supported Classes**: 6 Zones
    - **Model Accuracy**: 78%
    """)
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        {flaticon('tag', 18)}
        <h4 style="margin:0; font-size:1rem; font-weight:700;">Target Classes</h4>
    </div>
    """, unsafe_allow_html=True)
    for c_code, meta in CLASS_METADATA.items():
        color = meta['color']
        name = meta['display_name']
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:0.85rem;'>"
            f"<span style='display:inline-block; width:10px; height:10px; border-radius:50%; background-color:{color};'></span>"
            f"<span style='color:#e2e8f0; font-weight:500;'>{name}</span>"
            f"<span style='color:#64748b; font-size:0.75rem;'>({meta['severity']})</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        {flaticon('camera', 18)}
        <h4 style="margin:0; font-size:1rem; font-weight:700;">Photography Best Practices</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - **Angle**: Center frame on front or rear bumper/quarter.
    - **Lighting**: Ensure even daylight without extreme lens glare.
    - **Distance**: Stand 1.5 – 3 meters away from vehicle.
    - **Clarity**: Keep the damaged zone sharply in focus.
    """)
    
    if st.button("Reset App & Clear Cache", key="btn_reset_app", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:6px; font-size:0.76rem; color:#64748b;'>"
        f"{flaticon('info', 13)} Icons by <a href='https://www.flaticon.com/' target='_blank' style='color:#94a3b8; text-decoration:none;'>Flaticon</a>"
        f"</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Header Hero Section
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">{flaticon('lightning', 13, 'margin-right:2px;')} Neural Vision Triage</div>
    <h1 class="hero-title">Automated Vehicle Damage Inspector</h1>
    <p class="hero-subtitle">
        Upload or capture a photo of a vehicle's front or rear section. The fine-tuned ResNet-50 neural network will 
        classify the damage type, pinpoint the impact zone and gauge severity.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State for Selected Image
# ---------------------------------------------------------
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'image_source_name' not in st.session_state:
    st.session_state.image_source_name = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Sample Images Registry
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'samples')
SAMPLE_OPTIONS = [
    {"label": "sample_1", "filename": "front_crushed.jpg", "category": "F_Crushed", "desc": "Heavy front collision impact"},
    {"label": "sample_2", "filename": "front_breakage.jpg", "category": "F_Breakage", "desc": "Cracked bumper & headlight"},
    {"label": "sample_3", "filename": "front_normal.jpg", "category": "F_Normal", "desc": "Undamaged front exterior"},
    {"label": "sample_4", "filename": "rear_crushed.jpg", "category": "R_Crushed", "desc": "Severe rear tailgate smash"},
    {"label": "sample_5", "filename": "rear_breakage.jpg", "category": "R_Breakage", "desc": "Rear bumper / taillight fracture"},
    {"label": "sample_6", "filename": "rear_normal.jpg", "category": "R_Normal", "desc": "Clean undamaged rear end"}
]

# ---------------------------------------------------------
# Input Tabs: Upload / Sample Gallery
# ---------------------------------------------------------
tab_upload, tab_samples = st.tabs([
    "Upload Image",
    "Quick Sample Gallery"
])

with tab_upload:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        {flaticon('upload', 18)}
        <span style="font-weight:600; color:#cbd5e1; font-size:0.92rem;">Upload a vehicle photograph (JPG, JPEG, PNG, WEBP)</span>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a vehicle photograph",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Select a clear photo of the front or rear of the vehicle.",
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file).convert('RGB')
        st.session_state.image_source_name = uploaded_file.name

with tab_samples:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        {flaticon('gallery', 18)}
        <span style="font-weight:600; color:#cbd5e1; font-size:0.92rem;">Click any demo car image below to test the AI model instantly:</span>
    </div>
    """, unsafe_allow_html=True)
    sample_cols = st.columns(6)
    for idx, sample in enumerate(SAMPLE_OPTIONS):
        img_path = os.path.join(SAMPLE_DIR, sample["filename"])
        with sample_cols[idx]:
            if os.path.exists(img_path):
                thumb = Image.open(img_path)
                st.image(thumb, use_container_width=True)
                if st.button(sample["label"], key=f"sample_btn_{idx}", use_container_width=True):
                    st.session_state.current_image = Image.open(img_path).convert('RGB')
                    st.session_state.image_source_name = f"Sample: {sample['label']}"
                    st.session_state.uploader_key += 1
                    st.rerun()
            else:
                st.info(sample["label"])

# ---------------------------------------------------------
# Analysis & Results Section
# ---------------------------------------------------------
st.markdown("---")

if st.session_state.current_image is not None:
    active_img = st.session_state.current_image
    
    col_img, col_results = st.columns([5, 7], gap="large")
    
    with col_img:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            {flaticon('inspection', 22)}
            <h3 style="margin:0; font-size:1.35rem; font-weight:700;">Inspected Vehicle</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(active_img, caption=st.session_state.image_source_name, use_container_width=True)
        
        # Image Metadata Card
        width, height = active_img.size
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:10px 14px; font-size:0.83rem; color:#94a3b8; display:flex; justify-content:space-between; margin-bottom:12px;">
            <span style="display:flex; align-items:center; gap:6px;">{flaticon('resolution', 15)} <b>Resolution:</b> {width} × {height} px</span>
            <span style="display:flex; align-items:center; gap:6px;">{flaticon('folder', 15)} <b>Source:</b> {st.session_state.image_source_name}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Clear Current Image", key="btn_clear_image", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.image_source_name = None
            st.session_state.uploader_key += 1
            st.rerun()
    
    with col_results:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            {flaticon('report', 22)}
            <h3 style="margin:0; font-size:1.35rem; font-weight:700;">Diagnostic Assessment</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Analyzing neural network feature activations..."):
            result = predict_detailed(active_img)
            
        pred_class = result['class']
        conf = result['confidence']
        severity = result['severity']
        location = result['location']
        display_name = result['display_name']
        damage_type = result['damage_type']
        color = result['color']
        
        # Dynamic Severity Class & Flaticon Icons for Banner
        banner_class = "normal" if severity == "None" else ("severe" if severity == "Severe" else "moderate")
        if severity == "None":
            severity_icon_html = flaticon('normal_check', 44)
        elif severity == "Severe":
            severity_icon_html = flaticon('severe_alert', 44)
        else:
            severity_icon_html = flaticon('moderate_alert', 44)
        
        # Top Severity Banner
        st.markdown(f"""
        <div class="status-banner {banner_class}">
            <div style="display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                {severity_icon_html}
            </div>
            <div>
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; color:{color};">
                    {severity.upper()} SEVERITY ASSESSMENT
                </div>
                <div style="font-size:1.35rem; font-weight:800; color:#f8fafc; margin-top:2px;">
                    {display_name}
                </div>
                <div style="font-size:0.88rem; color:#cbd5e1; margin-top:2px;">
                    {result['description']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 4-KPI Metric Grid with Flaticon Icons
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display:flex; align-items:center; gap:6px;">
                    {flaticon('confidence', 13)} Confidence
                </div>
                <div class="metric-value" style="font-size:1.15rem; line-height:1.8; color:{color};">{conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display:flex; align-items:center; gap:6px;">
                    {flaticon('zone', 13)} Vehicle Zone
                </div>
                <div class="metric-value" style="font-size:1.15rem; line-height:1.8;">{location}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display:flex; align-items:center; gap:6px;">
                    {flaticon('condition', 13)} Condition
                </div>
                <div class="metric-value" style="font-size:1.15rem; line-height:1.8;">{damage_type.split('/')[0]}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display:flex; align-items:center; gap:6px;">
                    {flaticon('risk', 13)} Risk Level
                </div>
                <div class="metric-value" style="font-size:1.15rem; line-height:1.8; color:{color};">{severity}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        
        # Probability Breakdown Bars
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            {flaticon('target', 18)}
            <h4 style="margin:0; font-size:1.05rem; font-weight:700;">Classification Probability Distribution</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Sort probabilities descending
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        
        for c_name, p_val in sorted_probs:
            c_meta = CLASS_METADATA.get(c_name, {})
            bar_color = c_meta.get('color', '#3b82f6')
            c_display = c_meta.get('display_name', c_name)
            is_top = (c_name == pred_class)
            font_weight = "700" if is_top else "500"
            star = " ★" if is_top else ""
            
            st.markdown(f"""
            <div class="prob-bar-container">
                <div class="prob-label-row">
                    <span style="font-weight:{font_weight}; color:{'#ffffff' if is_top else '#94a3b8'};">
                        {c_display}{star}
                    </span>
                    <span style="font-weight:700; color:{'#ffffff' if is_top else '#94a3b8'};">
                        {p_val * 100:.1f}%
                    </span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {max(p_val * 100, 1.5)}%; background-color: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Empty state placeholder with Flaticon icon
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 14px; padding: 2.2rem; text-align: center; color: #94a3b8; margin-top: 1rem;">
        <div style="margin-bottom: 10px;">{flaticon('info', 36)}</div>
        <div style="font-weight: 700; font-size: 1.05rem; color: #e2e8f0; margin-bottom: 4px;">Ready for Vehicle Inspection</div>
        <div style="font-size: 0.88rem; color: #94a3b8;">Upload a vehicle photo above or select one of the Quick Samples to begin AI damage triage.</div>
    </div>
    """, unsafe_allow_html=True)
