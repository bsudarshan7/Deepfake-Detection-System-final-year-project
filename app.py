import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS (Red/Orange Theme)
# =========================
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .feature-card {
        background-color: #1e2127; padding: 25px; border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4); text-align: center;
        border: 1px solid #333; transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 20px;
    }
    .feature-card:hover { transform: translateY(-5px); border-color: #FF4500; }
    .feature-card h3 {
        background: -webkit-linear-gradient(#FF0000, #FFA500);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px; font-size: 1.2rem;
    }
    .feature-card p { color: #bfbfbf; font-size: 0.95rem; }
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #FF4500 !important; background-color: #1a1e24 !important;
        padding: 4rem 2rem !important; border-radius: 15px !important;
        transition: background-color 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover { background-color: #232830 !important; }
    .main-header {
        text-align: center; font-size: 3rem; font-weight: 800;
        background: -webkit-linear-gradient(#FF0000, #FFA500);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 1rem;
    }
    .sub-header { text-align: center; color: #a0a0a0; font-size: 1.2rem; margin-bottom: 3rem; }
    </style>
""", unsafe_allow_html=True)

# =========================
# GRAD-CAM IMPLEMENTATION
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, target_class]
        target.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam) 
        if torch.max(cam) != 0:
            cam -= torch.min(cam)
            cam /= torch.max(cam)
        return cam.cpu().numpy()

def apply_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_array = np.array(original_img)
    superimposed = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed)

# =========================
# CACHED MODEL LOADING
# =========================
@st.cache_resource
def load_model(model_path="deepfake_model.pth"):
    if not os.path.exists(model_path): return None, None, None
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model.load_state_dict(checkpoint['state_dict'])
        
        for param in model.parameters(): param.requires_grad = True 
        model.eval() 
        class_names = checkpoint.get('class_names', ['Class 0', 'Class 1'])
        grad_cam = GradCAM(model, model.features[-1])
        return model, class_names, grad_cam
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# =========================
# IMAGE TRANSFORMATION
# =========================
def transform_image(image):
    if image.mode != "RGB": image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).requires_grad_(True)

# =========================
# PREDICTION LOGIC
# =========================
def predict(model, grad_cam, image_tensor, original_img, class_names):
    with torch.set_grad_enabled(True):
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        heatmap_array = grad_cam.generate(image_tensor, predicted_idx.item())
        heatmap_img = apply_heatmap(original_img, heatmap_array)
    return class_names[predicted_idx.item()], confidence.item() * 100, heatmap_img

# =========================
# VIDEO PROCESSING
# =========================
def process_video(uploaded_file, model, grad_cam, class_names):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        os.unlink(tfile.name)
        return "Unknown", 0.0, []

    num_samples = min(8, frame_count) 
    step = max(1, frame_count // num_samples)
    class_votes, total_conf, frame_results = {}, {}, []
    
    try:
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = transform_image(pil_img)
            
            pred_class, conf, heatmap = predict(model, grad_cam, img_tensor, pil_img, class_names)
            class_votes[pred_class] = class_votes.get(pred_class, 0) + 1
            total_conf[pred_class] = total_conf.get(pred_class, 0.0) + conf
            
            frame_results.append({
                "frame": pil_img, "heatmap": heatmap,
                "prediction": pred_class, "confidence": conf
            })
            if len(frame_results) >= num_samples: break
    finally:
        cap.release()
        os.unlink(tfile.name)
    
    if not class_votes: return "Unknown", 0.0, []
    final_pred = max(class_votes, key=class_votes.get)
    final_conf = total_conf[final_pred] / class_votes[final_pred]
    return final_pred, final_conf, frame_results

# =========================
# MAIN APP UI
# =========================
def main():
    st.markdown('<div class="main-header">Deepfake Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an image or video to determine if it is Real or AI-Generated</div>', unsafe_allow_html=True)

    model, class_names, grad_cam = load_model("deepfake_model.pth")
    if model is None:
        st.error("🚨 **Model Not Found!**")
        st.warning("Please ensure `deepfake_model.pth` is in the same directory as this app.")
        st.stop()

    st.write("### 📤 Upload Media")
    uploaded_file = st.file_uploader("Drag and drop your image or video here", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        try:
            is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.write("#### Uploaded Subject")
                if is_video: st.video(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True, caption="Original Image")
                
            with res_col2:
                st.write("#### AI Analysis Results")
                with st.spinner("Analyzing media & generating thermal heatmaps..."):
                    if is_video:
                        prediction, confidence, frame_results = process_video(uploaded_file, model, grad_cam, class_names)
                    else:
                        image = Image.open(uploaded_file)
                        image_tensor = transform_image(image)
                        prediction, confidence, heatmap = predict(model, grad_cam, image_tensor, image, class_names)
                
                is_fake = any(keyword in prediction.lower() for keyword in ["fake", "spoof", "ai", "generated"])
                result_color = "#FF4500" if is_fake else "#4CAF50" 
                
                st.write("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="background-color: #1e2127; padding: 20px; border-radius: 10px; border-left: 5px solid {result_color};">
                        <h2 style="margin: 0; color: {result_color};">{prediction.upper()}</h2>
                        <p style="margin: 10px 0 5px 0; color: #aaa;">Overall Confidence Score: <strong>{confidence:.2f}%</strong></p>
                        <div style="width: 100%; background-color: #333; border-radius: 5px;">
                            <div style="width: {confidence}%; height: 10px; background-color: {result_color}; border-radius: 5px;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("<br>", unsafe_allow_html=True)
            st.write("### 🔬 Explainability Heatmaps")
            st.info("The red/orange areas indicate the specific pixels the AI focused on to make its decision.")
            
            if is_video:
                with st.expander("🎞️ View Extracted Video Frames & Heatmaps", expanded=True):
                    cols = st.columns(4)
                    for idx, res in enumerate(frame_results):
                        col = cols[idx % 4]
                        with col:
                            st.image(res['frame'], caption=f"Original Frame {idx+1}")
                            st.image(res['heatmap'], caption=f"{res['prediction']} ({res['confidence']:.1f}%)")
                            st.write("---")
            else:
                heat_col1, heat_col2 = st.columns(2)
                with heat_col1: st.image(image, caption="Original Image", use_container_width=True)
                with heat_col2: st.image(heatmap, caption="AI Focus Heatmap", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing the media: {e}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><h3>⚡ Fast Inference</h3><p>Powered by MobileNetV2 for lightning-fast analysis on the edge or server.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><h3>🧠 Visual Explainability</h3><p>Utilizes Grad-CAM heatmaps for both images and videos to show pixel-perfect logic.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><h3>🔒 Privacy First</h3><p>Media is processed strictly in-memory or temporarily and never stored.</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()