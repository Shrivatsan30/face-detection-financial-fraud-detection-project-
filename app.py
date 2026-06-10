from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import face_auth

app = Flask(__name__)

# Global shared camera
camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            camera.open(0)
            continue

        # Draw face boxes using InsightFace analyzer
        try:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces  = face_auth.analyzer.get(rgb)
            for face in faces:
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                score = round(float(face.det_score) * 100, 1)
                cv2.putText(frame, f"Face {score}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        except Exception:
            pass

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