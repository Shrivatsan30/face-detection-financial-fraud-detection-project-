from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import face_auth

app = Flask(__name__)

# Global shared camera instance
camera = cv2.VideoCapture(0)

def gen_frames():
    mp_face_detection = mp.solutions.face_detection
    frame_count = 0

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.6
    ) as detector:
        while True:
            success, frame = camera.read()
            if not success:
                camera.open(0)
                continue

            if frame_count % 3 == 0:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                if results.detections:
                    h, w = frame.shape[:2]
                    for det in results.detections:
                        box = det.location_data.relative_bounding_box
                        x1  = max(0, int(box.xmin * w))
                        y1  = max(0, int(box.ymin * h))
                        x2  = min(w, int((box.xmin + box.width) * w))
                        y2  = min(h, int((box.ymin + box.height) * h))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                        cv2.putText(frame, "Face Detected", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

            frame_count += 1
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/verify', methods=['POST'])
def verify():
    account = request.form.get('account', 'N/A')
    amount  = request.form.get('amount', 'N/A')

    result = face_auth.capture_and_compare()

    if isinstance(result, str):
        return jsonify({
            "authorized": False,
            "message":    result,
            "account":    account,
            "amount":     amount,
        })

    message = (
        f"{result['status']} | "
        f"Similarity: {result['similarity']}% | "
        f"Account: {account} | Amount: ₹{amount}"
    )

    return jsonify({
        "authorized": result["authorized"],
        "message":    message,
        "similarity": result["similarity"],
        "account":    account,
        "amount":     amount,
    })


if __name__ == '__main__':
    app.run(debug=True)