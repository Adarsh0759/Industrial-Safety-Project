import glob
from ultralytics import YOLO
import os

# Get all jpg files in the validation folder
image_paths = glob.glob('data/images/valid/*.jpg')
print(f"Found {len(image_paths)} images.")

# Load your YOLOv8 model
model = YOLO('yolov8n.pt')

# Run inference (returns a list of Results objects)
results = model(image_paths, batch=4)  # Adjust batch size as needed

# Create output directory for annotated images
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

for img_path, result in zip(image_paths, results):
    filename = os.path.basename(img_path)
    # Save the annotated image
    # If result.save() gives an error, use the OpenCV method below
    try:
        result.save(filename=os.path.join(output_dir, filename))
    except AttributeError:
        # Fallback for older Ultralytics versions
        import cv2
        annotated_img = result.plot()
        cv2.imwrite(os.path.join(output_dir, filename), annotated_img)

print("Batch processing complete! Check the 'outputs' folder.")


with open('detection_results.txt', 'w') as f:
    for idx, result in enumerate(results):
        summary = result.verbose()  # Or use str(result) if available
        f.write(f"{idx}: {summary}\n")
