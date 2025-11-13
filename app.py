import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL LOADING ---
# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def load_model():
    """Loads the YOLOv8n model from ultralytics."""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.error("Please ensure you have an internet connection to download the model.")
        return None

model = load_model()

# --- STREAMLIT UI ---
st.title("🤖 YOLOv8 Object Detection Web App")
st.markdown("Upload a video and see the real-time object detection in action.")

st.sidebar.header("Configuration")

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload a video file", 
    type=["mp4", "mov", "avi", "mkv"]
)

# Confidence Threshold Slider
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.5, 0.05
)

# Object Selection
if model:
    all_classes = model.names.values()
    default_classes = ['person', 'car', 'truck']
    
    selected_classes = st.sidebar.multiselect(
        "Select objects to detect",
        options=list(all_classes),
        default=default_classes
    )
    
    # Get the class IDs for the selected classes
    names_to_ids = {v: k for k, v in model.names.items()}
    target_class_ids = {names_to_ids[name] for name in selected_classes}
else:
    st.sidebar.warning("Model not loaded. Cannot select objects.")
    target_class_ids = set()

# Initialize session state for the stop button
if 'stop' not in st.session_state:
    st.session_state.stop = False

def stop_processing():
    st.session_state.stop = True

# --- VIDEO PROCESSING ---
if uploaded_file is not None and model:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    try:
        vid_cap = cv2.VideoCapture(video_path)
        if not vid_cap.isOpened():
            st.error(f"Error: Could not open video file.")
        else:
            # Placeholder for the video frames
            st_frame = st.empty()
            st.sidebar.button('Stop Processing', on_click=stop_processing)
            
            st.session_state.stop = False # Reset stop state at the beginning of processing

            while vid_cap.isOpened() and not st.session_state.stop:
                success, frame = vid_cap.read()
                if not success:
                    st.write("End of video file reached.")
                    break

                # Run YOLO model inference
                results = model(frame, verbose=False)
                detections = results[0].boxes

                # Draw bounding boxes
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Check if the detected object is a target and meets the confidence threshold
                    if class_id in target_class_ids and confidence > confidence_threshold:
                        box_color = (0, 255, 0) # Green
                        label = f'{model.names[class_id]} {confidence:.2f}'

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Convert the frame from BGR (OpenCV) to RGB (Streamlit)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                st_frame.image(frame_rgb, channels="RGB")

            # Clean up
            vid_cap.release()
            st.success("Video processing finished.")
            st.session_state.stop = False # Reset for the next run

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        st.session_state.stop = False # Ensure state is reset on error

else:
    st.info("Please upload a video file to begin.")