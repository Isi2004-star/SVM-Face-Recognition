import os
import cv2
import joblib
import numpy as np
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="SVM Face Recognition",
    page_icon="👤",
    layout="wide"
)

# Paths relative to this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_NAMES = {0: "Barack Obama", 1: "Donald Trump", 2: "George W Bush"}
MODEL_FILE = os.path.join(BASE_DIR, "SVM-Face Recognition.sav")
CASCADE_FILE = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")

@st.cache_resource
def load_resources():
    """Load and cache the trained pipeline and Haar Cascade classifier."""
    model = joblib.load(MODEL_FILE)
    
    # Try local XML file first, then fall back to OpenCV's built-in cascades
    if os.path.exists(CASCADE_FILE):
        cascade = cv2.CascadeClassifier(CASCADE_FILE)
    elif hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        cascade = cv2.CascadeClassifier(CASCADE_FILE)

    return model, cascade

def process_and_predict(image_bgr, model, cascade, min_neighbors=7, primary_only=False, min_confidence=0.3):
    """
    Detect faces, filter false positives, and predict identities with confidence.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    
    # Adaptive minimum face size (at least 60x60 or 8% of image dimension)
    min_dim = max(60, int(min(img_h, img_w) * 0.08))
    
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=min_neighbors,
        minSize=(min_dim, min_dim)
    )

    if len(faces) == 0:
        annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, []

    # Sort faces by area (width * height) descending
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    if primary_only:
        faces = faces[:1]

    results = []
    annotated_bgr = image_bgr.copy()

    for (x, y, w, h) in faces:
        cropped = gray[y:y + h, x:x + w]
        resized = cv2.resize(cropped, (50, 50))
        flattened = resized.reshape(1, -1).astype(np.float32) / 255.0

        # Predict class and calculate decision margin confidence
        pred_id = int(model.predict(flattened)[0])
        decision_scores = model.decision_function(flattened)[0]
        
        # Softmax approximation of decision scores for confidence display
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / np.sum(exp_scores)
        confidence = float(np.max(probabilities))

        if confidence < min_confidence:
            pred_name = "Uncertain / Noise"
            box_color = (0, 165, 255)  # Orange for uncertain
        else:
            pred_name = TARGET_NAMES.get(pred_id, "Unknown")
            box_color = (0, 200, 0)    # Green for confident

        results.append({
            "name": pred_name,
            "confidence": confidence,
            "face_crop": resized,
            "bbox": (x, y, w, h)
        })

        # Draw bounding box and label overlay
        cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), box_color, 3)
        label_text = f"{pred_name} ({confidence*100:.0f}%)" if pred_name != "Uncertain / Noise" else pred_name
        cv2.putText(
            annotated_bgr, label_text, (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2, cv2.LINE_AA
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results

# --- Main App Interface ---
st.title("👤 Real-Time SVM Face Recognition")
st.markdown(
    "Detect and classify faces of **Barack Obama**, **Donald Trump**, and **George W. Bush** "
    "using **Haar Cascades**, **PCA (Eigenfaces)**, and an **RBF-kernel Support Vector Machine**."
)

# Sidebar with model metrics and instructions
with st.sidebar:
    st.header("⚙️ Detection Settings")
    min_neighbors = st.slider(
        "Detector Strictness (minNeighbors):",
        min_value=3, max_value=12, value=7,
        help="Higher values eliminate false detections (hands, clothing folds, background noise)."
    )
    primary_only = st.checkbox(
        "Focus on Primary Face (Largest)",
        value=True,
        help="Recommended for single-person portrait photos to ignore background noise."
    )
    min_confidence = st.slider(
        "Confidence Threshold:",
        min_value=0.2, max_value=0.8, value=0.4, step=0.05,
        help="Filters out non-face noise patches where the SVM confidence is too low."
    )
    
    st.markdown("---")
    st.header("📊 Model Performance")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Test Accuracy", "85.3%")
    col_m2.metric("5-Fold CV", "83.0% ± 0.07")
    
    st.markdown("---")
    input_source = st.radio(
        "Choose Input Source:",
        ("Upload an Image", "Use Webcam", "Choose Sample Image")
    )

model, cascade = load_resources()
image_to_process = None

# Input Handling
if input_source == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif input_source == "Use Webcam":
    cam_file = st.camera_input("Take a snapshot with your webcam")
    if cam_file is not None:
        file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif input_source == "Choose Sample Image":
    if os.path.isdir(TEST_DATA_DIR):
        sample_files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        chosen = st.selectbox("Select a sample test image:", sample_files)
        if chosen:
            sample_path = os.path.join(TEST_DATA_DIR, chosen)
            image_to_process = cv2.imread(sample_path)

# Prediction and Output Display
if image_to_process is not None:
    annotated_image, detections = process_and_predict(
        image_to_process, model, cascade,
        min_neighbors=min_neighbors,
        primary_only=primary_only,
        min_confidence=min_confidence
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Detection & Classification Result")
        st.image(annotated_image, use_container_width=True)

    with col2:
        st.subheader("Identified Faces")
        if not detections:
            st.warning("No face detected. Try decreasing the detector strictness in the sidebar or upload a clearer frontal face.")
        else:
            for i, det in enumerate(detections, 1):
                if det["name"] == "Uncertain / Noise":
                    st.warning(f"**Region #{i}:** {det['name']} ({det['confidence']*100:.0f}% confidence)")
                else:
                    st.success(f"**Face #{i}:** {det['name']} ({det['confidence']*100:.0f}% confidence)")
                st.image(det["face_crop"], caption=f"{det['name']} (50×50 PCA input)", width=110)
else:
    st.info("👈 Please select or upload an image from the sidebar to run recognition.")
