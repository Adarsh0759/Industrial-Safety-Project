"""
MediaPipe Hand Gesture Recognition Module
Detects hand poses and recognizes hand signs and finger gestures using MediaPipe
Similar to: https://github.com/kinivi/hand-gesture-recognition-mediapipe
"""

import cv2
try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

import numpy as np
from enum import Enum

class HandGestureRecognizer:
    def __init__(self):
        """Initialize MediaPipe hand pose detection"""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available")
        
        self.hands = solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Hand gesture labels (basic hand signs)
        self.hand_labels = {
            0: "Open Palm",
            1: "Closed Fist", 
            2: "Pointing",
            3: "Thumbs Up",
            4: "Peace Sign",
            5: "Rock Sign",
            6: "OK Sign"
        }
        
        print("[MediaPipe] Hand gesture recognizer initialized")
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    
    def is_hand_open(self, landmarks):
        """Check if hand is open (palm facing camera)"""
        try:
            # Get key landmarks
            thumb_tip = landmarks[4]
            fingers_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            palm_center = landmarks[9]
            
            # Count extended fingers
            extended = 0
            for tip in fingers_tips:
                if tip.y < palm_center.y:  # Finger above palm
                    extended += 1
            
            return extended >= 3  # At least 3 fingers extended
        except:
            return False
    
    def is_fist_closed(self, landmarks):
        """Check if hand is in closed fist"""
        try:
            # Get key landmarks
            palm_center = landmarks[9]
            fingers_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            
            # Check if all fingers are below palm (closed)
            closed_count = 0
            for tip in fingers_tips:
                if tip.y > palm_center.y:
                    closed_count += 1
            
            return closed_count >= 3  # Most fingers closed
        except:
            return False
    
    def is_pointing(self, landmarks):
        """Check if hand is pointing (index finger extended, others closed)"""
        try:
            # Index finger tip and PIP
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            palm_center = landmarks[9]
            
            # Other fingers
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Index finger is extended
            index_extended = index_tip.y < index_pip.y
            
            # Other fingers are closed/curled
            others_closed = (middle_tip.y > palm_center.y and 
                           ring_tip.y > palm_center.y and 
                           pinky_tip.y > palm_center.y)
            
            return index_extended and others_closed
        except:
            return False
    
    def is_peace_sign(self, landmarks):
        """Check for peace sign (index and middle fingers extended, others closed)"""
        try:
            palm_center = landmarks[9]
            
            # Index and middle fingers extended
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            # Ring and pinky closed
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            extended = (index_tip.y < palm_center.y and middle_tip.y < palm_center.y)
            closed = (ring_tip.y > palm_center.y and pinky_tip.y > palm_center.y)
            
            return extended and closed
        except:
            return False
    
    def recognize_gesture(self, landmarks):
        """Recognize hand gesture from landmarks"""
        try:
            if self.is_fist_closed(landmarks):
                return 1, "Closed Fist"
            elif self.is_pointing(landmarks):
                return 2, "Pointing"
            elif self.is_peace_sign(landmarks):
                return 4, "Peace Sign"
            elif self.is_hand_open(landmarks):
                return 0, "Open Palm"
            else:
                return 0, "Unknown"
        except:
            return 0, "Error"
    
    def detect_gestures(self, frame):
        """
        Detect hand poses and recognize gestures in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            dict: {
                'frame': annotated frame,
                'gestures': list of detected gestures,
                'landmarks': raw MediaPipe landmarks
            }
        """
        h, w, c = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(frame_rgb)
        
        annotated_frame = frame.copy()
        gestures = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label (Left/Right)
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # Recognize gesture
                gesture_id, gesture_name = self.recognize_gesture(hand_landmarks.landmark)
                
                # Get hand bounding box
                landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                xs = [lm[0] for lm in landmark_list]
                ys = [lm[1] for lm in landmark_list]
                
                x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                y_min, y_max = int(min(ys) * h), int(max(ys) * h)
                
                # Add padding
                x_min = max(0, x_min - 10)
                y_min = max(0, y_min - 10)
                x_max = min(w, x_max + 10)
                y_max = min(h, y_max + 10)
                
                # Draw hand landmarks manually
                landmark_points = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark], dtype=np.int32)
                
                # Draw fingers connections (MediaPipe hand connections)
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
                ]
                
                for start, end in connections:
                    cv2.line(annotated_frame, tuple(landmark_points[start]), tuple(landmark_points[end]), 
                            (0, 255, 0), 1)
                
                # Draw landmarks as circles
                for pt in landmark_points:
                    cv2.circle(annotated_frame, tuple(pt), 3, (0, 255, 0), -1)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Draw label
                label_text = f"{hand_label}: {gesture_name} ({confidence:.2f})"
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x_min, y_min - 25), 
                            (x_min + label_size[0], y_min), color, -1)
                cv2.putText(annotated_frame, label_text, (x_min, y_min - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Store gesture info
                gestures.append({
                    'hand': hand_label,
                    'gesture': gesture_name,
                    'gesture_id': gesture_id,
                    'confidence': float(confidence),
                    'bbox': [x_min, y_min, x_max, y_max],
                    'landmarks': hand_landmarks.landmark
                })
        
        return {
            'frame': annotated_frame,
            'gestures': gestures,
            'handedness': results.multi_handedness if results.multi_handedness else None
        }
    
    def get_hand_keypoints(self, hand_landmarks):
        """Extract keypoints from hand landmarks (for ML training)"""
        keypoints = []
        for landmark in hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return keypoints
    
    def release(self):
        """Release resources"""
        self.hands.close()
