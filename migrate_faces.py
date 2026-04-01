import os
import cv2
import glob

DATASET_DIR = "dataset"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for img_path in glob.glob(os.path.join(DATASET_DIR, "*", "face.jpg")):
    frame = cv2.imread(img_path)
    if frame is None: 
        continue
    
    # Try detecting
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"No face found in {img_path}, skipping crop.")
        continue
        
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    
    face_crop = frame[y1:y2, x1:x2]
    cv2.imwrite(img_path, face_crop)
    print(f"Cropped {img_path}")

# Remove existing PKL files
for pkl in glob.glob(os.path.join(DATASET_DIR, "*.pkl")):
    os.remove(pkl)
    print(f"Deleted {pkl}")
