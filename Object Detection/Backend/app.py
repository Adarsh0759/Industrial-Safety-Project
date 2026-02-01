from flask import Flask, Response, render_template
import cv2
from detector import SafetyDetector

app = Flask(__name__)
detector = SafetyDetector()

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        processed_frame = detector.detect_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
