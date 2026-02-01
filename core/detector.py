import cv2
import os
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import YOLO
from ultralytics import YOLO

# Import MediaPipe gesture recognizer
try:
    from .mediapipe_gestures import HandGestureRecognizer
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not available - gesture recognition disabled")

class SafetyDetector:
    def __init__(self):
        print("=" * 70)
        print("Initializing SafetyDetector - MULTI-MODEL DETECTION SYSTEM")

        print("=" * 70)
        
        # Initialize MediaPipe Hand Gesture Recognizer
        print("\n[MediaPipe] Initializing MediaPipe Hand Gesture Recognizer...")
        self.mediapipe_gestures = None
        self.has_mediapipe = False
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mediapipe_gestures = HandGestureRecognizer()
                self.has_mediapipe = True
                print("✓ MediaPipe Hand Gesture Recognizer loaded!")
            except Exception as e:
                print(f"⚠ MediaPipe initialization failed: {e}")
        
        # Load pretrained YOLOv8m model for object detection (MAIN)
        print("\n[1/4] Loading YOLOv8m Model (General Object Detection)...")
        
        try:
            self.detection_model = YOLO('yolov8m.pt')
            print("✓ YOLOv8m model loaded successfully!")
        except Exception as e:
            print(f"✗ Model load failed: {e}")
            raise RuntimeError(f"Failed to load YOLOv8m model: {e}")
        
        # Load hand gesture classification model (ASL A-Z, 0-9)
        print("\n[2/4] Loading Hand Gesture Classification Model...")
        gesture_model_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'hand.pt')
        
        self.gesture_model = None
        self.has_gesture_model = False
        
        if os.path.exists(gesture_model_path):
            try:
                self.gesture_model = YOLO(gesture_model_path)
                num_classes = len(self.gesture_model.names)
                print(f"✓ Hand gesture model loaded! ({num_classes} gesture classes: ASL A-Z, 0-9)")
                self.has_gesture_model = True
            except Exception as e:
                print(f"⚠ Hand gesture model loading failed: {e}")
        else:
            print(f"⚠ Hand gesture model not found: {gesture_model_path}")
        
        # Load Detection Model 1 (last.pt - User's trained model)
        print("\n[3/4] Loading Detection Model 1 (last.pt)...")
        ppe_model1_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'last.pt')
        
        self.ppe_model1 = None
        self.has_ppe_model1 = False
        
        if os.path.exists(ppe_model1_path):
            try:
                self.ppe_model1 = YOLO(ppe_model1_path)
                num_classes = len(self.ppe_model1.names)
                print(f"✓ Model 1 (last.pt) loaded! ({num_classes} classes)")
                self.has_ppe_model1 = True
            except Exception as e:
                print(f"⚠ PPE Model 1 loading failed: {e}")
        else:
            print(f"⚠ PPE Model 1 not found: {ppe_model1_path}")
        
        # Load Detection Model 2 (best.pt - User's trained model)
        print("\n[4/4] Loading Detection Model 2 (best.pt)...")
        ppe_model2_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'best.pt')
        
        self.ppe_model2 = None
        self.has_ppe_model2 = False
        
        if os.path.exists(ppe_model2_path):
            try:
                self.ppe_model2 = YOLO(ppe_model2_path)
                num_classes = len(self.ppe_model2.names)
                print(f"✓ Model 2 (best.pt) loaded! ({num_classes} classes)")
                self.has_ppe_model2 = True
            except Exception as e:
                print(f"⚠ PPE Model 2 loading failed: {e}")
        else:
            print(f"⚠ PPE Model 2 not found: {ppe_model2_path}")
        
        # Load Detection Model 3 (best (1).pt - User's trained model)
        print("\n[5/5] Loading Detection Model 3 (best (1).pt)...")
        extra_model_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'best (1).pt')
        
        self.extra_model = None
        self.has_extra_model = False
        
        if os.path.exists(extra_model_path):
            try:
                self.extra_model = YOLO(extra_model_path)
                num_classes = len(self.extra_model.names)
                print(f"✓ Model 3 (best (1).pt) loaded! ({num_classes} classes)")
                self.has_extra_model = True
            except Exception as e:
                print(f"⚠ Model 3 loading failed: {e}")
        else:
            print(f"⚠ Model 3 not found: {extra_model_path}")
        
        # Load Hand Detection Model (hand.pt - for hand-specific detection)
        print("\n[6/6] Loading Hand Detection Model (hand.pt)...")
        hand_detect_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'hand.pt')
        
        self.hand_detect_model = None
        self.has_hand_detect = False
        
        if os.path.exists(hand_detect_path):
            try:
                self.hand_detect_model = YOLO(hand_detect_path)
                num_classes = len(self.hand_detect_model.names)
                print(f"✓ Hand Detection Model loaded! ({num_classes} classes)")
                self.has_hand_detect = True
            except Exception as e:
                print(f"⚠ Hand Detection Model loading failed: {e}")
        else:
            print(f"⚠ Hand Detection Model not found: {hand_detect_path}")
        
        # Disable heavy pose models (they cause lag)
        self.pose_model = None
        self.nano_pose_model = None
        
        print("\n" + "=" * 70)
        print("✓ SafetyDetector Initialized - MULTI-MODEL SYSTEM (7 MODELS)")
        print(f"  - Object Detection: YOLOv8m (80 COCO classes)")
        print(f"  - MediaPipe Hand Gestures: {'ENABLED' if self.has_mediapipe else 'DISABLED'}")
        print(f"  - Hand Gestures (YOLO): {'ENABLED' if self.has_gesture_model else 'DISABLED'}")
        print(f"  - Detection Model 1 (last.pt): {'ENABLED' if self.has_ppe_model1 else 'DISABLED'}")
        print(f"  - Detection Model 2 (best.pt): {'ENABLED' if self.has_ppe_model2 else 'DISABLED'}")
        print(f"  - Detection Model 3 (best (1).pt): {'ENABLED' if self.has_extra_model else 'DISABLED'}")
        print(f"  - Hand Detection Model: {'ENABLED' if self.has_hand_detect else 'DISABLED'}")
        print(f"  - Body Pose: DISABLED (for speed)")
        print("=" * 70)
        print("\n⚡ Running in OPTIMIZED 6-model mode for maximum detection coverage\n")
        
        # Confidence threshold
        self.conf = 0.45
        
        # COCO class mapping (80 classes)
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
            14: 'cat', 15: 'dog', 16: 'horse', 17: 'sheep', 18: 'cow', 19: 'elephant',
            20: 'bear', 21: 'zebra', 22: 'giraffe', 23: 'backpack', 24: 'umbrella', 
            25: 'handbag', 26: 'tie', 27: 'suitcase', 28: 'frisbee', 29: 'skis',
            30: 'snowboard', 31: 'sports ball', 32: 'kite', 33: 'baseball bat', 
            34: 'baseball glove', 35: 'skateboard', 36: 'surfboard', 37: 'tennis racket', 
            38: 'bottle', 39: 'wine glass', 40: 'cup', 41: 'fork', 42: 'knife', 
            43: 'spoon', 44: 'bowl', 45: 'banana', 46: 'apple', 47: 'sandwich', 
            48: 'orange', 49: 'broccoli', 50: 'carrot', 51: 'hot dog', 52: 'pizza', 
            53: 'donut', 54: 'cake', 55: 'chair', 56: 'couch', 57: 'potted plant', 
            58: 'bed', 59: 'dining table', 60: 'toilet', 61: 'tv', 62: 'laptop', 
            63: 'mouse', 64: 'remote', 65: 'keyboard', 66: 'microwave', 67: 'oven', 
            68: 'toaster', 69: 'sink', 70: 'refrigerator', 71: 'book', 72: 'clock', 
            73: 'vase', 74: 'scissors', 75: 'teddy bear', 76: 'hair drier', 77: 'toothbrush', 78: 'helmet'
        }
    
    def detect_frame(self, frame):
        """
        Detect objects and hand gestures - OPTIMIZED (fast).
        """
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        
        detections = {
            'hardhats': 0,
            'people': 0,
            'vehicles': 0,
            'backpacks': 0,
            'hand_gestures': 0,
            'body_gestures': 0,
            'gesture_details': [],
            'objects': []
        }
        
        # MediaPipe Hand Gesture Detection
        if self.has_mediapipe and self.mediapipe_gestures is not None:
            try:
                mediapipe_results = self.mediapipe_gestures.detect_gestures(frame)
                annotated_frame = mediapipe_results['frame']
                
                if mediapipe_results['gestures']:
                    for gesture in mediapipe_results['gestures']:
                        detections['hand_gestures'] += 1
                        detections['gesture_details'].append({
                            'type': 'mediapipe',
                            'hand': gesture['hand'],
                            'gesture': gesture['gesture'],
                            'confidence': gesture['confidence'],
                            'bbox': gesture['bbox']
                        })
            except Exception as e:
                pass
        
        try:
            # Object Detection (MAIN)
            results = self.detection_model.predict(
                source=frame,
                conf=self.conf,
                verbose=False,
                device='cpu',
                imgsz=416  # Smaller for speed
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        class_name = self.coco_classes.get(cls_id, f'Object {cls_id}')
                        
                        if class_name == 'person':
                            detections['people'] += 1
                        elif class_name == 'helmet':
                            detections['hardhats'] += 1
                        elif class_name == 'backpack':
                            detections['backpacks'] += 1
                        elif class_name in ['bus', 'truck', 'train', 'car', 'motorcycle', 'airplane']:
                            detections['vehicles'] += 1
                        
                        color = self._get_color(class_name)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name} {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 23), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 8),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        detections['objects'].append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        except Exception as e:
            pass
        
        # Hand Gesture Detection - Scan frame for hands and classify gestures
        if self.has_gesture_model and self.gesture_model is not None:
            try:
                # Scan multiple regions of the frame for hand gestures
                h, w = frame.shape[:2]
                
                # Define scan regions (left, center, right)
                regions = [
                    (0, 0, w//3, h),  # Left third
                    (w//3, 0, 2*w//3, h),  # Center third
                    (2*w//3, 0, w, h),  # Right third
                ]
                
                for x1, y1, x2, y2 in regions:
                    region = frame[y1:y2, x1:x2]
                    
                    if region.size == 0:
                        continue
                    
                    try:
                        gesture_results = self.gesture_model.predict(
                            source=region,
                            conf=0.4,
                            verbose=False,
                            device='cpu',
                            imgsz=416
                        )
                        
                        if gesture_results and len(gesture_results) > 0:
                            gesture_result = gesture_results[0]
                            
                            if gesture_result.boxes is not None and len(gesture_result.boxes) > 0:
                                for box in gesture_result.boxes:
                                    cls_id = int(box.cls[0].item())
                                    conf = float(box.conf[0].item())
                                    
                                    # Convert box coordinates from region to full frame
                                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                                    frame_x1 = x1 + bx1
                                    frame_y1 = y1 + by1
                                    frame_x2 = x1 + bx2
                                    frame_y2 = y1 + by2
                                    
                                    gesture_name = self.gesture_model.names.get(cls_id, f'Gesture {cls_id}')
                                    
                                    detections['hand_gestures'] += 1
                                    detections['gesture_details'].append({
                                        'gesture': gesture_name,
                                        'type': 'hand',
                                        'confidence': conf
                                    })
                                    
                                    color = (200, 0, 200)
                                    cv2.rectangle(annotated_frame, (frame_x1, frame_y1), (frame_x2, frame_y2), color, 2)
                                    
                                    gesture_label = f"{gesture_name} {conf:.2f}"
                                    label_size, _ = cv2.getTextSize(gesture_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(annotated_frame, (frame_x1, frame_y1 - 25), 
                                                (frame_x1 + label_size[0], frame_y1), color, -1)
                                    cv2.putText(annotated_frame, gesture_label, (frame_x1, frame_y1 - 8),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    except Exception as e:
                        pass
                
            except Exception as e:
                pass
        
        # PPE Detection Model 1 (last.pt)
        if self.has_ppe_model1 and self.ppe_model1 is not None:
            try:
                ppe_results1 = self.ppe_model1.predict(
                    source=frame,
                    conf=0.4,
                    verbose=False,
                    device='cpu',
                    imgsz=416
                )
                
                if ppe_results1 and len(ppe_results1) > 0:
                    ppe_result1 = ppe_results1[0]
                    
                    if ppe_result1.boxes is not None and len(ppe_result1.boxes) > 0:
                        for box in ppe_result1.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            ppe_class = self.ppe_model1.names.get(cls_id, f'PPE_{cls_id}')
                            
                            # Draw PPE detection
                            color = (0, 165, 255)  # Orange
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            ppe_label = f"PPE: {ppe_class}"
                            label_size, _ = cv2.getTextSize(ppe_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(annotated_frame, (x1, y1 - 20), 
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(annotated_frame, ppe_label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            detections['objects'].append({
                                'class': f'PPE_{ppe_class}',
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'model': 'ppe_last'
                            })
            
            except Exception as e:
                pass
        
        # PPE Detection Model 2 (best.pt)
        if self.has_ppe_model2 and self.ppe_model2 is not None:
            try:
                ppe_results2 = self.ppe_model2.predict(
                    source=frame,
                    conf=0.4,
                    verbose=False,
                    device='cpu',
                    imgsz=416
                )
                
                if ppe_results2 and len(ppe_results2) > 0:
                    ppe_result2 = ppe_results2[0]
                    
                    if ppe_result2.boxes is not None and len(ppe_result2.boxes) > 0:
                        for box in ppe_result2.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            ppe_class = self.ppe_model2.names.get(cls_id, f'PPE_{cls_id}')
                            
                            # Draw PPE detection
                            color = (0, 128, 255)  # Darker orange
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            ppe_label = f"PPE: {ppe_class}"
                            label_size, _ = cv2.getTextSize(ppe_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(annotated_frame, (x1, y1 - 20), 
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(annotated_frame, ppe_label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            detections['objects'].append({
                                'class': f'PPE_{ppe_class}',
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'model': 'ppe_best'
                            })
            
            except Exception as e:
                pass
        
        # Detection Model 3 (best (1).pt)
        if self.has_extra_model and self.extra_model is not None:
            try:
                extra_results = self.extra_model.predict(
                    source=frame,
                    conf=0.4,
                    verbose=False,
                    device='cpu',
                    imgsz=416
                )
                
                if extra_results and len(extra_results) > 0:
                    extra_result = extra_results[0]
                    
                    if extra_result.boxes is not None and len(extra_result.boxes) > 0:
                        for box in extra_result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            extra_class = self.extra_model.names.get(cls_id, f'Object_{cls_id}')
                            
                            # Draw detection with blue color
                            color = (255, 0, 0)  # Blue
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            extra_label = f"Model3: {extra_class}"
                            label_size, _ = cv2.getTextSize(extra_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(annotated_frame, (x1, y1 - 20), 
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(annotated_frame, extra_label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            detections['objects'].append({
                                'class': extra_class,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'model': 'best_1'
                            })
            
            except Exception as e:
                pass
        
        # Hand Detection Model (hand.pt)
        if self.has_hand_detect and self.hand_detect_model is not None:
            try:
                hand_results = self.hand_detect_model.predict(
                    source=frame,
                    conf=0.4,
                    verbose=False,
                    device='cpu',
                    imgsz=416
                )
                
                if hand_results and len(hand_results) > 0:
                    hand_result = hand_results[0]
                    
                    if hand_result.boxes is not None and len(hand_result.boxes) > 0:
                        for box in hand_result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            hand_class = self.hand_detect_model.names.get(cls_id, f'Hand_{cls_id}')
                            
                            # Draw hand detection with magenta color
                            color = (255, 0, 255)  # Magenta
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            hand_label = f"Hand: {hand_class}"
                            label_size, _ = cv2.getTextSize(hand_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(annotated_frame, (x1, y1 - 20), 
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(annotated_frame, hand_label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            detections['objects'].append({
                                'class': f'Hand_{hand_class}',
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'model': 'hand_detect'
                            })
            
            except Exception as e:
                pass
        
        return {
            'frame': annotated_frame,
            'detections': detections,
            'hardhats': detections['hardhats'],
            'people': detections['people'],
            'vehicles': detections['vehicles'],
            'backpacks': detections['backpacks'],
            'hand_gestures': detections['hand_gestures'],
            'body_gestures': 0,
            'objects': detections['objects']
        }
    
    def _get_color(self, class_name):
        """Return BGR color based on class name"""
        colors = {
            'helmet': (0, 255, 0),
            'person': (0, 255, 0),
            'car': (255, 0, 0),
            'truck': (255, 0, 0),
            'bus': (255, 0, 0),
            'train': (255, 0, 0),
            'motorcycle': (255, 0, 0),
            'airplane': (255, 0, 0),
            'backpack': (255, 255, 0),
        }
        return colors.get(class_name, (200, 200, 200))
