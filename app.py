from flask import Flask, render_template, Response, request
import cv2
import face_auth  # your backend file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Streaming video feed
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/verify', methods=['POST'])
def verify():
    account = request.form['account']
    amount = request.form['amount']
    verified = face_auth.capture_and_compare()
    if verified:
        return f"✅ Transaction Approved | Account: {account}, Amount: {amount}"
    else:
        return "❌ Transaction Denied | Face not recognized."

if __name__ == '__main__':
    app.run(debug=True)
