import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp

# ── Config ──────────────────────────────────────────────────────────────────
REFERENCE_DIR        = os.path.join("static", "reference_faces")
CAPTURED_PATH        = os.path.join("dataset", "champ", "test.jpg")
SIMILARITY_THRESHOLD = 0.82
FACE_SIZE            = (128, 128)

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_detection = mp.solutions.face_detection


# ── Helpers ──────────────────────────────────────────────────────────────────

def preprocess(img):
    """CLAHE on luminance channel for lighting invariance."""
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab     = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def extract_face(image_bgr):
    """
    Detect face using MediaPipe, crop and return normalised
    grayscale patch. Returns None if no face found.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.6
    ) as detector:
        results = detector.process(rgb)

    if not results.detections:
        return None

    best = max(results.detections, key=lambda d: d.score[0])
    box  = best.location_data.relative_bounding_box
    h, w = image_bgr.shape[:2]

    x1 = max(0, int(box.xmin * w))
    y1 = max(0, int(box.ymin * h))
    x2 = min(w, int((box.xmin + box.width)  * w))
    y2 = min(h, int((box.ymin + box.height) * h))

    face = image_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, FACE_SIZE)
    face = face.astype(np.float32) / 255.0
    return face


def get_embedding(face_patch):
    """Flatten normalized face patch into a feature vector."""
    return face_patch.flatten().reshape(1, -1)


def load_reference_embeddings():
    if not os.path.exists(REFERENCE_DIR):
        raise FileNotFoundError(f"Reference folder not found: {REFERENCE_DIR}")

    embeddings = []
    for fname in os.listdir(REFERENCE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(REFERENCE_DIR, fname))
        if img is None:
            continue
        img  = preprocess(img)
        face = extract_face(img)
        if face is not None:
            embeddings.append(get_embedding(face))
        else:
            print(f"[WARN] No face detected in reference: {fname}")

    if not embeddings:
        raise ValueError("No valid faces found in reference_faces folder.")

    return embeddings


# ── Main ─────────────────────────────────────────────────────────────────────

def capture_and_compare():
    os.makedirs(os.path.dirname(CAPTURED_PATH), exist_ok=True)

    # Load references
    try:
        ref_embeddings = load_reference_embeddings()
    except (FileNotFoundError, ValueError) as e:
        return {"authorized": False, "status": f"❌ Enrollment error: {e}",
                "similarity": 0, "distance": 1}

    # Use the shared global camera from app.py
    try:
        from app import camera
        frame = None
        for _ in range(5):
            ret, f = camera.read()
            if ret:
                frame = f
    except Exception:
        # Fallback: open camera independently
        cam = cv2.VideoCapture(0)
        frame = None
        for _ in range(5):
            ret, f = cam.read()
            if ret:
                frame = f
        cam.release()

    if frame is None:
        return {"authorized": False, "status": "❌ Failed to capture frame",
                "similarity": 0, "distance": 1}

    # Preprocess + detect
    frame     = preprocess(frame)
    cv2.imwrite(CAPTURED_PATH, frame)

    live_face = extract_face(frame)
    if live_face is None:
        return {"authorized": False,
                "status": "❌ No face detected — look directly at camera",
                "similarity": 0, "distance": 1}

    live_emb   = get_embedding(live_face)
    scores     = [cosine_similarity(live_emb, ref)[0][0] for ref in ref_embeddings]
    best_score = float(np.max(scores))
    authorized = best_score >= SIMILARITY_THRESHOLD
    result     = "✅ Authorized" if authorized else "❌ Blocked"

    return {
        "authorized": authorized,
        "status":     result,
        "similarity": round(best_score * 100, 1),
        "distance":   round(1 - best_score, 4),
        "refs_used":  len(ref_embeddings),
    }


if __name__ == "__main__":
    print(capture_and_compare())