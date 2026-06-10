import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# ── Config ──────────────────────────────────────────────────────────────────
REFERENCE_DIR        = os.path.join("static", "reference_faces")
CAPTURED_PATH        = os.path.join("dataset", "champ", "test.jpg")
SIMILARITY_THRESHOLD = 0.45  # cosine distance, lower = stricter

# ── InsightFace setup ────────────────────────────────────────────────────────
# First run downloads ArcFace model (~140MB) automatically
analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
analyzer.prepare(ctx_id=0, det_size=(640, 640))


# ── Helpers ──────────────────────────────────────────────────────────────────

def preprocess(img):
    """CLAHE on luminance channel for lighting invariance."""
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab     = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_embedding(img_bgr):
    """
    Run InsightFace on image, return 512-d ArcFace embedding.
    Returns None if no face detected.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces   = analyzer.get(img_rgb)
    if not faces:
        return None
    # Pick face with highest detection score
    best = max(faces, key=lambda f: f.det_score)
    return best.embedding  # 512-d numpy array


def cosine_distance(a, b):
    """Lower = more similar. 0.0 = identical."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1 - np.dot(a, b))


def load_reference_embeddings():
    if not os.path.exists(REFERENCE_DIR):
        raise FileNotFoundError(f"Reference folder not found: {REFERENCE_DIR}")

    embeddings = []
    for fname in sorted(os.listdir(REFERENCE_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(REFERENCE_DIR, fname))
        if img is None:
            continue
        img = preprocess(img)
        emb = get_embedding(img)
        if emb is not None:
            embeddings.append(emb)
            print(f"[INFO] Loaded reference: {fname}")
        else:
            print(f"[WARN] No face in reference: {fname}")

    if not embeddings:
        raise ValueError("No valid faces found in reference_faces folder.")

    return embeddings


# ── Main ─────────────────────────────────────────────────────────────────────

def capture_and_compare():
    os.makedirs(os.path.dirname(CAPTURED_PATH), exist_ok=True)

    # Load reference embeddings
    try:
        ref_embeddings = load_reference_embeddings()
    except (FileNotFoundError, ValueError) as e:
        return {"authorized": False, "status": f"❌ Enrollment error: {e}",
                "similarity": 0, "distance": 1}

    # Grab frame from shared camera
    try:
        from app import camera
        frame = None
        for _ in range(5):
            ret, f = camera.read()
            if ret:
                frame = f
    except Exception:
        cam   = cv2.VideoCapture(0)
        frame = None
        for _ in range(5):
            ret, f = cam.read()
            if ret:
                frame = f
        cam.release()

    if frame is None:
        return {"authorized": False, "status": "❌ Failed to capture frame",
                "similarity": 0, "distance": 1}

    # Preprocess and save
    frame = preprocess(frame)
    cv2.imwrite(CAPTURED_PATH, frame)

    # Get live embedding
    live_emb = get_embedding(frame)
    if live_emb is None:
        return {"authorized": False,
                "status": "❌ No face detected — look directly at camera",
                "similarity": 0, "distance": 1}

    # Compare against all references
    distances  = [cosine_distance(live_emb, ref) for ref in ref_embeddings]
    best_dist  = float(np.min(distances))
    similarity = round((1 - best_dist) * 100, 1)
    authorized = best_dist <= SIMILARITY_THRESHOLD
    status     = "✅ Authorized" if authorized else "❌ Blocked"

    return {
        "authorized": authorized,
        "status":     status,
        "similarity": similarity,
        "distance":   round(best_dist, 4),
        "refs_used":  len(ref_embeddings),
    }


if __name__ == "__main__":
    print(capture_and_compare())