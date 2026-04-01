"""
Recognition Engine for Smart Door Lock System
===============================================
Course: Computer Vision & Robotics Intelligence

This module implements TWO core Computer Vision tasks:

1. FACE RECOGNITION (Identity Verification)
   - Uses DeepFace with VGG-Face pre-trained model (Transfer Learning)
   - VGG-Face was trained on 2.6 million face images by Oxford Visual Geometry Group
   - We use it for "one-shot learning" - only 1 reference image per student needed
   - Comparison method: Cosine Similarity between face embeddings (128-d vectors)

2. EMOTION DETECTION (Facial Expression Classification)
   - Uses our CUSTOM TRAINED CNN model on FER-2013 dataset
   - Falls back to DeepFace emotion analysis if custom model not available
   - 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

Data Storage:
   - Face images: ./dataset/<studentid_name>/face.jpg (JPEG, captured from webcam)
   - Student records: ./smartdoor.db (SQLite database)
   - Trained emotion model: ./trained_models/emotion_model.keras
"""

import os
import cv2
import numpy as np
from deepface import DeepFace

# ─── Paths ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.keras")
CUSTOM_MODEL_H5_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")

os.makedirs(DATASET_DIR, exist_ok=True)

# ─── Emotion labels (must match training order) ────────────────

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ─── Load custom emotion model if available ─────────────────────

_custom_emotion_model = None
_using_custom_model = False


def _load_custom_model():
    """Load our custom-trained emotion detection CNN model."""
    global _custom_emotion_model, _using_custom_model
    
    if _custom_emotion_model is not None:
        return _custom_emotion_model
    
    model_path = None
    if os.path.exists(CUSTOM_MODEL_PATH):
        model_path = CUSTOM_MODEL_PATH
    elif os.path.exists(CUSTOM_MODEL_H5_PATH):
        model_path = CUSTOM_MODEL_H5_PATH
    
    if model_path:
        try:
            from tensorflow import keras
            _custom_emotion_model = keras.models.load_model(model_path)
            _using_custom_model = True
            print(f"✅ Custom emotion model loaded from: {model_path}")
            return _custom_emotion_model
        except Exception as e:
            print(f"⚠️  Failed to load custom model: {e}")
            print("   Falling back to DeepFace emotion analysis.")
            _using_custom_model = False
    else:
        print("ℹ️  No custom emotion model found. Using DeepFace.")
        print(f"   Train one with: python train_emotion_model.py")
        _using_custom_model = False
    
    return None


def _predict_emotion_custom(face_gray_48):
    """
    Predict emotion using our custom-trained CNN.
    
    Input: Grayscale face image, resized to 48x48
    Process:
      1. Normalize pixel values to [0, 1]
      2. Reshape to (1, 48, 48, 1) for batch prediction
      3. Model outputs softmax probabilities for 7 emotions
      4. Return dominant emotion and all probabilities
    """
    model = _load_custom_model()
    if model is None:
        return None, None
    
    # Preprocess: normalize and reshape
    img = face_gray_48.astype('float32') / 255.0
    img = img.reshape(1, 48, 48, 1)
    
    # Predict
    predictions = model.predict(img, verbose=0)[0]
    
    # Build emotion dict
    emotions = {}
    for i, label in enumerate(EMOTION_LABELS):
        emotions[label] = float(predictions[i]) * 100  # Convert to percentage
    
    dominant = EMOTION_LABELS[np.argmax(predictions)]
    
    return dominant, emotions


# ═══════════════════════════════════════════════════════════════════
# FACE IMAGE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def save_face_image(student_id, name, image_data):
    """
    Save a captured face image for a student.
    
    Storage structure:
      dataset/
      ├── STU001_John_Doe/
      │   └── face.jpg          ← Reference face for recognition
      ├── STU002_Jane_Smith/
      │   └── face.jpg
      └── ...
    
    Args:
        student_id: Unique student ID (e.g., "STU001")
        name: Full name (e.g., "John Doe")
        image_data: numpy array (BGR from OpenCV) or bytes from webcam
    
    Returns:
        Saved file path, or None on failure
    """
    # Create student folder with format: StudentID_Name
    safe_name = f"{student_id}_{name.replace(' ', '_')}"
    student_dir = os.path.join(DATASET_DIR, safe_name)
    os.makedirs(student_dir, exist_ok=True)

    file_path = os.path.join(student_dir, "face.jpg")

    if isinstance(image_data, np.ndarray):
        cv2.imwrite(file_path, image_data)
    elif isinstance(image_data, bytes):
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(file_path, img)
        else:
            return None
    else:
        return None

    print(f"📸 Face saved: {file_path}")
    return file_path


# ═══════════════════════════════════════════════════════════════════
# FACE RECOGNITION + EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════════

def recognize_face(frame):
    """
    Main recognition pipeline - OPTIMIZED FOR SPEED.
    
    STEP 1: Fast Face Detection on downscaled image using OpenCV
    STEP 2: Crop the original high-res face
    STEP 3: Emotion Detection on the crop
    STEP 4: Identity match via DeepFace (detector_backend='skip' to avoid double-detection)
    """
    if frame is None or frame.size == 0:
        return None

    _load_custom_model()

    # ─── STEP 1: Fast Face Detection ──────────────
    # Downscale for much faster Haar Cascade
    h, w = frame.shape[:2]
    max_w = 320
    if w > max_w:
        ratio = max_w / float(w)
        small_frame = cv2.resize(frame, (max_w, int(h * ratio)))
    else:
        small_frame = frame
        ratio = 1.0

    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # More relaxed parameters to catch faces easily
    faces = face_cascade.detectMultiScale(
        gray_small, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None
    
    # Take the largest face
    x_s, y_s, w_s, h_s = max(faces, key=lambda f: f[2] * f[3])
    
    # Map coordinates back to original image scale
    x = int(x_s / ratio)
    y = int(y_s / ratio)
    fw = int(w_s / ratio)
    fh = int(h_s / ratio)
    
    # Pad the crop slightly for better DeepFace/Emotion results
    pad_x = int(fw * 0.1)
    pad_y = int(fh * 0.1)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)

    face_crop = frame[y1:y2, x1:x2]
    face_region = {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1}
    
    if face_crop.size == 0:
        return None

    # ─── STEP 2: Emotion Detection ──────────────────────────────
    dominant_emotion = "neutral"
    emotion_data = {}
    
    if _using_custom_model:
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray_crop, (48, 48))
        dominant_emotion, emotion_data = _predict_emotion_custom(face_resized)
        if dominant_emotion is None:
            dominant_emotion = "neutral"
            emotion_data = {}
    else:
        try:
            # Tell DeepFace to skip detection since we already cropped it
            analyses = DeepFace.analyze(
                face_crop,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            if analyses:
                analysis = analyses[0] if isinstance(analyses, list) else analyses
                emotion_data = analysis.get("emotion", {})
                dominant_emotion = analysis.get("dominant_emotion", "neutral")
        except Exception:
            pass

    # ─── STEP 3: Face Recognition (Identity Matching) ───────────
    registered_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    if not registered_folders:
        return {
            "recognized": False,
            "student_folder": None,
            "distance": None,
            "emotion": dominant_emotion,
            "emotions": emotion_data,
            "face_region": face_region,
        }

    try:
        # We pass the FULL FRAME and use "opencv" detector so DeepFace does PROPER alignment
        # (Alignment dramatically increases VGG-Face accuracy!)
        results = DeepFace.find(
            img_path=frame,
            db_path=DATASET_DIR,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
    except Exception as e:
        print(f"⚠️  DeepFace.find error: {e}")
        return {
            "recognized": False,
            "student_folder": None,
            "distance": None,
            "emotion": dominant_emotion,
            "emotions": emotion_data,
            "face_region": face_region,
        }

    # Check results
    if results is not None and len(results) > 0:
        df = results[0]
        if not df.empty:
            best_match_path = df.iloc[0]["identity"]
            # Extract folder name: dataset/STU001_John/face.jpg → STU001_John
            rel = os.path.relpath(best_match_path, DATASET_DIR)
            student_folder = rel.split(os.sep)[0]

            # Get similarity distance
            distance_col = [c for c in df.columns if "distance" in c.lower() or "cos" in c.lower()]
            dist_val = float(df.iloc[0][distance_col[0]]) if distance_col else 0.0

            return {
                "recognized": True,
                "student_folder": student_folder,
                "distance": dist_val,
                "emotion": dominant_emotion,
                "emotions": emotion_data,
                "face_region": face_region,
            }

    return {
        "recognized": False,
        "student_folder": None,
        "distance": None,
        "emotion": dominant_emotion,
        "emotions": emotion_data,
        "face_region": face_region,
    }


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_emotion_emoji(emotion):
    """Return an emoji for a given emotion."""
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "surprise": "😲",
        "fear": "😨",
        "disgust": "🤢",
        "neutral": "😐",
    }
    return emoji_map.get(emotion, "❓")


def get_emotion_label_id(emotion):
    """Return a human-friendly Indonesian label for an emotion."""
    label_map = {
        "happy": "Senang",
        "sad": "Sedih",
        "angry": "Marah",
        "surprise": "Terkejut",
        "fear": "Cemas / Takut",
        "disgust": "Jijik",
        "neutral": "Netral",
    }
    return label_map.get(emotion, emotion)


def get_model_info():
    """Return info about which models are being used."""
    _load_custom_model()
    return {
        "face_recognition": "VGG-Face (Transfer Learning, Oxford VGG Group)",
        "emotion_detection": "Custom CNN (FER-2013)" if _using_custom_model else "DeepFace Built-in",
        "custom_model_loaded": _using_custom_model,
        "custom_model_path": CUSTOM_MODEL_PATH if _using_custom_model else None,
        "dataset_dir": DATASET_DIR,
        "registered_faces": len([
            f for f in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, f))
        ]) if os.path.exists(DATASET_DIR) else 0,
    }
