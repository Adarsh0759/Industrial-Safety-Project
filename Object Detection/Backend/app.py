from pathlib import Path
from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import queue
from core.detector import SafetyDetector


class FPSCounter:
    def __init__(self):
        self.frame_times = []
        self.fps = 0

    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) - 1

    def get_fps(self):
        return self.fps


BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'Templates'),
    static_folder=str(BASE_DIR / 'static')
)
CORS(app)

frame_queue = queue.Queue(maxsize=1)
detection_data = {
    'hardhats': 0,
    'vests': 0,
    'people': 0,
    'vehicles': 0,
    'backpacks': 0,
    'hand_gestures': 0,
    'gesture_details': [],
    'detections': [],
    'models_active': 0,
    'fps': 0,
    'timestamp': 0
}
detection_lock = threading.Lock()
last_frame = None
detector = None
capture_thread = None
capture_started = False
detector_thread = None
detector_start_requested = False


def load_detector():
    global detector
    try:
        print("Initializing detection models...")
        loaded_detector = SafetyDetector()
        detector = loaded_detector
        with detection_lock:
            detection_data['models_active'] = len(loaded_detector.ppe_models) + (1 if loaded_detector.has_gesture_model else 0)
        print("Detection models loaded successfully")
    except Exception as e:
        print(f"Detector initialization failed: {e}")
        detector = None


def ensure_detector():
    global detector_thread, detector_start_requested
    if detector is None and not detector_start_requested:
        detector_start_requested = True
        detector_thread = threading.Thread(target=load_detector, daemon=True)
        detector_thread.start()
    return detector


def ensure_capture():
    global capture_thread, capture_started
    if capture_started:
        return

    capture_started = True
    capture_thread = threading.Thread(target=video_capture_thread, daemon=True)
    capture_thread.start()
    time.sleep(0.2)


def video_capture_thread():
    global last_frame
    try:
        cap = None

        # On Windows, DirectShow is usually the least noisy backend for webcams.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            try:
                candidate = cv2.VideoCapture(0, backend)
            except TypeError:
                candidate = cv2.VideoCapture(0)

            if candidate.isOpened():
                cap = candidate
                break

            candidate.release()

        if cap is None:
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
                except Exception:
                    pass
                try:
                    frame_queue.put_nowait(frame)
                except Exception:
                    pass

            time.sleep(0.01)

        cap.release()
    except Exception as e:
        print(f"Camera error: {e}")


def generate_frames():
    fps_counter = FPSCounter()
    ensure_detector()
    ensure_capture()

    while True:
        try:
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
                    cv2.putText(frame, "Waiting for camera...", (140, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            active_detector = detector
            if active_detector is not None:
                try:
                    result = active_detector.detect_frame(frame)
                    frame = result['frame']

                    with detection_lock:
                        detection_data['hardhats'] = result['hardhats']
                        detection_data['vests'] = result['vests']
                        detection_data['people'] = result['people']
                        detection_data['vehicles'] = result['vehicles']
                        detection_data['backpacks'] = result['backpacks']
                        detection_data['hand_gestures'] = result['hand_gestures']
                        detection_data['gesture_details'] = result['detections'].get('gesture_details', [])
                        detection_data['detections'] = result['objects']
                        detection_data['models_active'] = result['detections'].get('models_active', 0)
                        detection_data['timestamp'] = time.time()
                except Exception as e:
                    print(f"Detection error: {e}")

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

            time.sleep(0.01)
        except Exception as e:
            print(f"Frame error: {e}")
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    ensure_detector()
    ensure_capture()
    with detection_lock:
        return jsonify({
            'hardhats': detection_data['hardhats'],
            'vests': detection_data['vests'],
            'people': detection_data['people'],
            'vehicles': detection_data['vehicles'],
            'backpacks': detection_data['backpacks'],
            'hand_gestures': detection_data['hand_gestures'],
            'gesture_details': detection_data['gesture_details'][:10],
            'models_active': detection_data['models_active'],
            'fps': round(detection_data['fps'], 1)
        })


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'service': 'Safety Detection API'})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SAFETY MONITOR DASHBOARD")
    print("=" * 60)
    print("Dashboard: http://localhost:5000")
    print("API Stats: http://localhost:5000/api/stats")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
