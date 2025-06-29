import cv2
from ultralytics import YOLO

class SafetyDetector:
    def __init__(self):
        self.ppe_model = YOLO('D:/VISION/Object Detection/Backend/Models/my.torchscript', task='detect')
        self.gesture_model = YOLO('D:/VISION/Object Detection/Backend/Models/hand.torchscript', task='detect')
    
    def detect_frame(self, frame):
        # Initialize with default values
        gesture_results = None
        ppe_results = None
        
        try:
            # Detect PPE
            ppe_results = self.ppe_model(frame)
            annotated_frame = ppe_results[0].plot() if ppe_results else frame.copy()
        except Exception as e:
            print(f"PPE detection error: {e}")
            annotated_frame = frame.copy()
            ppe_results = None

        try:
            # Detect gestures
            gesture_results = self.gesture_model(frame)
        except Exception as e:
            print(f"Gesture detection error: {e}")
            gesture_results = None

        # Debugging prints
        print(f"PPE boxes: {len(ppe_results[0].boxes) if ppe_results and ppe_results[0] else 0}")
        print(f"Gesture boxes: {len(gesture_results[0].boxes) if gesture_results and gesture_results[0] else 0}")

        # Draw gesture results if available
        if gesture_results and gesture_results[0] and gesture_results[0].boxes:
            for box in gesture_results[0].boxes:
                if box.conf[0] < 0.5:  # Confidence threshold
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.gesture_model.names[int(box.cls[0])]
                confidence = box.conf[0].item()  # Convert tensor to float
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, 
                           f"{label} {confidence:.2f}",
                           (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
