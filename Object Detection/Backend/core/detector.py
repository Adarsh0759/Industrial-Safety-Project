import os
import warnings

import cv2
import torch
from ultralytics import YOLO

warnings.filterwarnings('ignore')

_ORIGINAL_TORCH_LOAD = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _trusted_torch_load


class SafetyDetector:
    def __init__(self):
        self.models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Models'))
        self.ppe_conf = 0.4
        self.gesture_conf = 0.35

        print("=" * 70)
        print("Initializing SafetyDetector - LOCAL MULTI-MODEL MODE")
        print("=" * 70)

        self.ppe_models = []
        self.gesture_model = self._load_model('hand.pt', 'Hand gesture model')
        self.has_gesture_model = self.gesture_model is not None

        for filename, label in [
            ('last.pt', 'PPE model 1'),
            ('best.pt', 'PPE model 2'),
            ('best (1).pt', 'PPE model 3'),
        ]:
            model = self._load_model(filename, label)
            if model is not None:
                self.ppe_models.append((filename, model))

        print("\n" + "=" * 70)
        print("SafetyDetector initialized")
        print(f"  - PPE models active: {len(self.ppe_models)}")
        print(f"  - Hand gesture model: {'ENABLED' if self.has_gesture_model else 'DISABLED'}")
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
            'models_active': len(self.ppe_models) + (1 if self.has_gesture_model else 0)
        }

        for model_name, model in self.ppe_models:
            annotated_frame = self._run_ppe_model(frame, annotated_frame, detections, model_name, model)

        if self.has_gesture_model:
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

    def _run_ppe_model(self, frame, annotated_frame, detections, model_name, model):
        try:
            results = model.predict(
                source=frame,
                conf=self.ppe_conf,
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
                class_name = model.names.get(cls_id, f'Object_{cls_id}')

                self._accumulate_ppe_counts(detections, class_name)

                color = self._get_color(class_name)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label = f"{model_name}: {class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + label_size[0], y1), color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )

                detections['objects'].append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'model': model_name
                })
        except Exception:
            pass

        return annotated_frame

    def _run_gesture_model(self, frame, annotated_frame, detections):
        try:
            results = self.gesture_model.predict(
                source=frame,
                conf=self.gesture_conf,
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

                color = (195, 48, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label = f"hand.pt: {gesture_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + label_size[0], y1), color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )

                detections['objects'].append({
                    'class': gesture_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'model': 'hand.pt'
                })
        except Exception:
            pass

        return annotated_frame

    def _normalize_label(self, class_name):
        return str(class_name).strip().lower().replace('_', ' ').replace('-', ' ')

    def _accumulate_ppe_counts(self, detections, class_name):
        normalized = self._normalize_label(class_name)

        if normalized == 'person':
            detections['people'] += 1
        elif normalized in ['hardhat', 'hard hat', 'helmet']:
            detections['hardhats'] += 1
        elif 'vest' in normalized:
            detections['vests'] += 1
        elif normalized in ['vehicle', 'car', 'truck', 'bus']:
            detections['vehicles'] += 1
        elif 'backpack' in normalized:
            detections['backpacks'] += 1

    def _get_color(self, class_name):
        normalized = self._normalize_label(class_name)
        if 'hardhat' in normalized or 'helmet' in normalized:
            return (0, 200, 0)
        if 'vest' in normalized:
            return (0, 180, 255)
        if normalized == 'person':
            return (30, 220, 30)
        if normalized.startswith('no '):
            return (0, 0, 255)
        if 'vehicle' in normalized:
            return (255, 130, 0)
        return (200, 200, 200)
