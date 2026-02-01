from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import queue
from core.detector import SafetyDetector

app = Flask(__name__)
CORS(app)

# Global variables
frame_queue = queue.Queue(maxsize=1)
detection_data = {
    'hardhats': 0,
    'people': 0,
    'vehicles': 0,
    'backpacks': 0,
    'hand_gestures': 0,
    'body_gestures': 0,
    'gesture_details': [],
    'pose_keypoints': [],
    'detections': [],
    'fps': 0,
    'timestamp': 0
}
detection_lock = threading.Lock()
last_frame = None
detector = None

def video_capture_thread():
    """Thread for capturing frames from camera"""
    global last_frame
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Webcam not available - generating test frames")
            frame_num = 0
            while True:
                frame = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "TEST FRAME", (200, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_num}", (150, 300), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 1)
                last_frame = frame
                frame_num += 1
                time.sleep(0.033)
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            last_frame = frame
            
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except:
                    pass
                try:
                    frame_queue.put_nowait(frame)
                except:
                    pass
            
            time.sleep(0.01)
        
        cap.release()
    except Exception as e:
        print(f"Camera error: {e}")

def generate_frames():
    """Generate MJPEG stream with real-time detection"""
    global detector
    fps_counter = FPSCounter()
    frame_num = 0
    
    # Initialize detector on first use
    if detector is None:
        try:
            print("Initializing detection models...")
            detector = SafetyDetector()
            print("Detection models loaded successfully")
        except Exception as e:
            detector = None
    
    while True:
        try:
            # Get frame from queue or use last frame
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
                    cv2.putText(frame, "Waiting for camera...", (140, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Run detection if available
            if detector is not None:
                try:
                    result = detector.detect_frame(frame)
                    frame = result['frame']
                    
                    with detection_lock:
                        detection_data['hardhats'] = result['hardhats']
                        detection_data['people'] = result['people']
                        detection_data['vehicles'] = result['vehicles']
                        detection_data['backpacks'] = result['backpacks']
                        detection_data['hand_gestures'] = result['hand_gestures']
                        detection_data['gesture_details'] = result['detections'].get('gesture_details', [])
                        detection_data['detections'] = result['objects']
                        detection_data['timestamp'] = time.time()
                except Exception as e:
                    pass
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                frame_bytes = buffer.tobytes()
                fps_counter.update()
                
                with detection_lock:
                    detection_data['fps'] = fps_counter.get_fps()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
                
                frame_num += 1
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Frame error: {e}")
            time.sleep(0.1)

class FPSCounter:
    def __init__(self, window=30):
        self.window = window
        self.timestamps = []
    
    def update(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])

# Start capture thread
capture_thread = threading.Thread(target=video_capture_thread, daemon=True)
capture_thread.start()
time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    with detection_lock:
        return jsonify({
            'hardhats': detection_data['hardhats'],
            'people': detection_data['people'],
            'vehicles': detection_data['vehicles'],
            'backpacks': detection_data['backpacks'],
            'hand_gestures': detection_data['hand_gestures'],
            'body_gestures': detection_data['body_gestures'],
            'gesture_details': detection_data['gesture_details'][:10],
            'fps': round(detection_data['fps'], 1)
        })

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'service': 'Safety Detection API'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SAFETY MONITOR DASHBOARD")
    print("="*60)
    print("Dashboard: http://localhost:5000")
    print("API Stats: http://localhost:5000/api/stats")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
