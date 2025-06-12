from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train15/weights/best.pt")

# Path to your input video
video_path = "samples/test_video (2).mp4"

# Run inference and track frames
results = model(video_path, stream=True, save=True)

# Initialize frame counter
frame_count = 0

for result in results:
    frame_count += 1
    boxes = result.boxes
    classes = boxes.cls  # Detected class indices
    confs = boxes.conf  # Confidence scores
    
    # Print frame number and detections
    print(f"Frame {frame_count}: Detected {len(classes)} objects")
    
    # Optional: Print class names and confidences
    class_names = [model.names[int(cls)] for cls in classes]
    print(f"Classes: {class_names}")
    print(f"Confidences: {confs.numpy().round(2)}\n")

print("Inference complete! Output saved to 'runs/detect/predict'")
