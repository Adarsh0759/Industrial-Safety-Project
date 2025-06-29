from ultralytics import YOLO

# Load your original model
model = YOLO('D:/VISION/Object Detection/Backend/Models/best.pt')

# Re-export with current PyTorch version
model.export(format='torchscript')  # Creates 'my.torchscript'
