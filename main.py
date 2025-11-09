

# --- Step 1: Import necessary libraries ---
import cv2
from ultralytics import YOLO
import numpy as np

# --- Step 2: Initialize the AI Model and Video ---

# Load the pre-trained YOLOv8 'nano' model for fast performance
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure you have an internet connection to download the model.")
    exit()

# IMPORTANT: SPECIFY THE PATH TO YOUR TEST VIDEO FILE HERE
video_path = '/Users/charan/Desktop/projects/intrusion detection/testvid.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# --- Step 3: Define the Target Objects ---

# Define the object classes we want to monitor.
# We use class IDs from the COCO dataset. For example:
# 0 is 'person', 2 is 'car', 7 is 'truck'
# To see all possible objects and their IDs, you can print model.names
# print(model.names)
TARGET_CLASSES = {0, 2, 7}  # Currently monitoring for people, cars, and trucks

# --- Step 4: Main Processing Loop ---

print("Starting Object Detection on Video...")
print("Press 'q' in the video window to quit.")

while True:
    # Read a new frame from the video
    success, frame = cap.read()
    if not success:
        print("End of video file reached.")
        break

    # --- AI Inference ---
    # Run the YOLO model on the current frame
    results = model(frame, verbose=False) # Set verbose=False for cleaner output
    
    # Get the list of detected objects (boxes)
    detections = results[0].boxes

    # --- Visualization Logic ---
    
    # Loop through each detected object in the frame
    for box in detections:
        # Extract information from the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = int(box.cls[0])             # Object's class ID
        confidence = float(box.conf[0])        # Detection confidence score

        # Check if the detected object is one of our targets and has high confidence
        if class_id in TARGET_CLASSES and confidence > 0.5:
            
            # Set the color for the bounding box (here, a bright green)
            box_color = (0, 255, 0)
            
            # Create the label text (e.g., "person 0.89")
            label = f'{model.names[class_id]} {confidence:.2f}'

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw the label text above the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # --- Display the final processed frame ---
    cv2.imshow("Object Detection in Video", frame)

    # Check for user input to quit the program (press 'q')
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Step 5: Clean Up ---
print("Shutting down system.")
cap.release()
cv2.destroyAllWindows()