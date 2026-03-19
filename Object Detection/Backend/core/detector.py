import os
import warnings

import cv2
import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings('ignore')

try:
    from .mediapipe_gestures import HandGestureRecognizer
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - gesture recognition disabled")


_ORIGINAL_TORCH_LOAD = torch.load


def _trusted_torch_load(*args, **kwargs):
    """
    Ultralytics checkpoints used by this project were created before
    PyTorch 2.6 changed the default torch.load behavior.
    These are local trusted files, so opt back into weights_only=False.
    """
    kwargs.setdefault('weights_only', False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _trusted_torch_load


class SafetyDetector:
    def __init__(self):
        self.models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Models'))
        self.conf = 0.45

        print("=" * 70)
        print("Initializing SafetyDetector - MULTI-MODEL DETECTION SYSTEM")
        print("=" * 70)

        self.mediapipe_gestures = None
        self.has_mediapipe = False
        if MEDIAPIPE_AVAILABLE:
            try:
                print("\n[MediaPipe] Initializing MediaPipe Hand Gesture Recognizer...")
                self.mediapipe_gestures = HandGestureRecognizer()
                self.has_mediapipe = True
                print("MediaPipe Hand Gesture Recognizer loaded")
            except Exception as exc:
                print(f"MediaPipe initialization failed: {exc}")

        self.detection_model = self._load_model('yolov8m.pt', 'General object detection')
        self.gesture_model = self._load_model('hand.pt', 'Hand gesture model')
        self.ppe_model1 = self._load_model('last.pt', 'PPE model 1')
        self.ppe_model2 = self._load_model('best.pt', 'PPE model 2')
        self.extra_model = self._load_model('best (1).pt', 'PPE model 3')

        self.has_detection_model = self.detection_model is not None
        self.has_gesture_model = self.gesture_model is not None
        self.has_ppe_model1 = self.ppe_model1 is not None
        self.has_ppe_model2 = self.ppe_model2 is not None
        self.has_extra_model = self.extra_model is not None

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
            73: 'vase', 74: 'scissors', 75: 'teddy bear', 76: 'hair drier',
            77: 'toothbrush', 78: 'helmet'
        }

        print("\n" + "=" * 70)
        print("SafetyDetector initialized")
        print(f"  - General object detection: {'ENABLED' if self.has_detection_model else 'DISABLED'}")
        print(f"  - MediaPipe hand gestures: {'ENABLED' if self.has_mediapipe else 'DISABLED'}")
        print(f"  - Hand gesture model: {'ENABLED' if self.has_gesture_model else 'DISABLED'}")
        print(f"  - PPE model 1: {'ENABLED' if self.has_ppe_model1 else 'DISABLED'}")
        print(f"  - PPE model 2: {'ENABLED' if self.has_ppe_model2 else 'DISABLED'}")
        print(f"  - PPE model 3: {'ENABLED' if self.has_extra_model else 'DISABLED'}")
        print("=" * 70)

    def _load_model(self, filename, label):
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            print(f"{label} not found: {path}")
            return None

        try:
            model = YOLO(path)
            print(f"{label} loaded from {path}")
            return model
        except Exception as exc:
            print(f"{label} failed to load from {path}: {exc}")
            return None

    def detect_frame(self, frame):
        annotated_frame = frame.copy()
        detections = {
            'hardhats': 0,
            'vests': 0,
            'people': 0,
            'vehicles': 0,
            'backpacks': 0,
            'hand_gestures': 0,
            'body_gestures': 0,
            'gesture_details': [],
            'objects': [],
            'violations': {
                'no_hardhat': 0,
                'no_vest': 0
            }
        }

        if self.has_mediapipe and self.mediapipe_gestures is not None:
            try:
                mediapipe_results = self.mediapipe_gestures.detect_gestures(frame)
                annotated_frame = mediapipe_results['frame']
                for gesture in mediapipe_results.get('gestures', []):
                    detections['hand_gestures'] += 1
                    detections['gesture_details'].append({
                        'type': 'mediapipe',
                        'hand': gesture['hand'],
                        'gesture': gesture['gesture'],
                        'confidence': gesture['confidence'],
                        'bbox': gesture['bbox']
                    })
            except Exception:
                pass

        if self.has_detection_model and self.detection_model is not None:
            annotated_frame = self._run_detection_model(
                model=self.detection_model,
                frame=frame,
                annotated_frame=annotated_frame,
                detections=detections,
                label_prefix='',
                model_key='coco',
                class_name_resolver=lambda cls_id, model: self.coco_classes.get(cls_id, f'Object {cls_id}')
            )

        if self.has_ppe_model1 and self.ppe_model1 is not None:
            annotated_frame = self._run_detection_model(
                model=self.ppe_model1,
                frame=frame,
                annotated_frame=annotated_frame,
                detections=detections,
                label_prefix='PPE',
                model_key='ppe_last'
            )

        if self.has_ppe_model2 and self.ppe_model2 is not None:
            annotated_frame = self._run_detection_model(
                model=self.ppe_model2,
                frame=frame,
                annotated_frame=annotated_frame,
                detections=detections,
                label_prefix='PPE',
                model_key='ppe_best'
            )

        if self.has_extra_model and self.extra_model is not None:
            annotated_frame = self._run_detection_model(
                model=self.extra_model,
                frame=frame,
                annotated_frame=annotated_frame,
                detections=detections,
                label_prefix='PPE',
                model_key='ppe_extra'
            )

        if self.has_gesture_model and self.gesture_model is not None:
            annotated_frame = self._run_gesture_model(frame, annotated_frame, detections)

        return {
            'frame': annotated_frame,
            'detections': detections,
            'hardhats': detections['hardhats'],
            'vests': detections['vests'],
            'people': detections['people'],
            'vehicles': detections['vehicles'],
            'backpacks': detections['backpacks'],
            'hand_gestures': detections['hand_gestures'],
            'body_gestures': 0,
            'objects': detections['objects']
        }

    def _run_detection_model(
        self,
        model,
        frame,
        annotated_frame,
        detections,
        label_prefix,
        model_key,
        class_name_resolver=None
    ):
        try:
            results = model.predict(
                source=frame,
                conf=0.4,
                verbose=False,
                device='cpu',
                imgsz=416
            )
            if not results:
                return annotated_frame

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return annotated_frame

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if class_name_resolver is not None:
                    class_name = class_name_resolver(cls_id, model)
                else:
                    class_name = model.names.get(cls_id, f'Object_{cls_id}')

                self._accumulate_detection_count(detections, class_name)

                color = self._get_color(class_name)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                rendered_label = f"{label_prefix}: {class_name}" if label_prefix else class_name
                text = f"{rendered_label} {conf:.2f}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + text_size[0], y1), color, -1)
                cv2.putText(
                    annotated_frame,
                    text,
                    (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

                detections['objects'].append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'model': model_key
                })
        except Exception:
            pass

        return annotated_frame

    def _run_gesture_model(self, frame, annotated_frame, detections):
        try:
            results = self.gesture_model.predict(
                source=frame,
                conf=0.35,
                verbose=False,
                device='cpu',
                imgsz=416
            )
            if not results:
                return annotated_frame

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return annotated_frame

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                gesture_name = self.gesture_model.names.get(cls_id, f'Gesture {cls_id}')

                detections['hand_gestures'] += 1
                detections['gesture_details'].append({
                    'gesture': gesture_name,
                    'type': 'hand',
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

                color = (200, 0, 200)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                text = f"{gesture_name} {conf:.2f}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + text_size[0], y1), color, -1)
                cv2.putText(
                    annotated_frame,
                    text,
                    (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )
        except Exception:
            pass

        return annotated_frame

    def _get_color(self, class_name):
        normalized = self._normalize_label(class_name)
        if 'hardhat' in normalized or 'helmet' in normalized:
            return (0, 200, 0)
        if 'vest' in normalized:
            return (0, 180, 255)
        if normalized == 'person':
            return (0, 255, 0)
        if normalized in ['car', 'truck', 'bus', 'train', 'motorcycle', 'airplane', 'vehicle']:
            return (255, 0, 0)
        if 'backpack' in normalized:
            return (255, 255, 0)
        if 'no hardhat' in normalized or 'no safety vest' in normalized:
            return (0, 0, 255)
        return (200, 200, 200)

    def _normalize_label(self, class_name):
        return str(class_name).strip().lower().replace('_', ' ').replace('-', ' ')

    def _accumulate_detection_count(self, detections, class_name):
        normalized = self._normalize_label(class_name)

        if normalized == 'person':
            detections['people'] += 1
        elif normalized in ['vehicle', 'car', 'truck', 'bus', 'train', 'motorcycle', 'airplane']:
            detections['vehicles'] += 1
        elif normalized in ['backpack', 'bagpack']:
            detections['backpacks'] += 1
        elif normalized in ['hardhat', 'hard hat', 'helmet']:
            detections['hardhats'] += 1
        elif 'vest' in normalized:
            detections['vests'] += 1
        elif normalized == 'no hardhat':
            detections['violations']['no_hardhat'] += 1
        elif normalized == 'no safety vest':
            detections['violations']['no_vest'] += 1
