import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def capture_and_compare():
    dataset = "dataset"
    name = "champ"
    path = os.path.join(dataset, name)
    (width, height) = (130, 100)

    os.makedirs(path, exist_ok=True)

    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return "❌ Could not access the webcam"

    count = 1
    captured_face = None

    while count <= 1:
        ret, img = cam.read()
        if not ret:
            cam.release()
            return "❌ Failed to capture image from webcam"

        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

        for (x, y, w, h) in faces:
            faceOnly = grayImg[y:y+h, x:x+w]
            resizeImg = cv2.resize(faceOnly, (width, height))
            captured_face = resizeImg
            cv2.imwrite(os.path.join(path, "test.jpg"), resizeImg)
            count += 1
        if cv2.waitKey(10) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    ref_path = os.path.join("static", "reference.jpg")
    if not os.path.exists(ref_path):
        return f"❌ Reference image not found at {ref_path}"

    reference_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if reference_img is None:
        return f"❌ Failed to load reference image from {ref_path}"

    reference_img = cv2.resize(reference_img, (width, height))

    captured_flat = captured_face.flatten().reshape(1, -1)
    reference_flat = reference_img.flatten().reshape(1, -1)
    similarity = cosine_similarity(captured_flat, reference_flat)[0][0]

    threshold = 0.85
    result = "✅ Authorized" if similarity > threshold else "❌ Blocked"

    return f"Transaction {result}\nSimilarity: {similarity:.2f}"

# ✅ Call the function to execute
if __name__ == "__main__":
    output = capture_and_compare()
    print(output)