from ultralytics import YOLO

# 1. LOAD YOUR ACTUAL HAND GESTURE MODEL
model = YOLO('D:/VISION/Object Detection/Backend/Models/hand.pt')  # Must be your actual .pt model file

# 2. EXPORT TO TORCHSCRIPT
model.export(
    format='torchscript',
    name='models/hand',  # Saves as models/hand.torchscript
    imgsz=640,  # Match your training size
    device='cpu'  # Use 'cuda' if GPU available
)

print("Successfully exported hand.torchscript to models/ folder")
