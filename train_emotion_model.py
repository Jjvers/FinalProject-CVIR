"""
Emotion Detection Model Training Script (Transfer Learning + Custom CNN)
=========================================================================
Course: Computer Vision & Robotics Intelligence

This script trains an emotion detection model using TWO approaches:
  1. Custom CNN trained from scratch on FER-2013
  2. Transfer Learning using MobileNetV2 (pretrained on ImageNet 1.4M images)
     + Fine-tuned on FER-2013 for higher accuracy

Dataset: FER-2013 (Facial Expression Recognition 2013)
  - Source: Kaggle (https://www.kaggle.com/datasets/msambare/fer2013)
  - 35,887 grayscale 48x48 face images
  - 7 emotion classes:
      0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
  - Train: 28,709 images | Test: 7,178 images

Output Files (saved in training_results/ and trained_models/):
  - emotion_model.keras              → Trained model (Keras format)
  - emotion_model.h5                 → Trained model (H5 format)
  - confusion_matrix.png             → Confusion matrix heatmap
  - training_history.png             → Accuracy & Loss curves
  - sample_predictions.png           → Grid of test predictions
  - classification_report.txt        → Per-class precision/recall/F1
  - dataset_samples.png              → Sample images from each class
  - model_info.txt                   → Architecture & hyperparameter details

Requirements:
  pip install tensorflow opencv-python pandas scikit-learn matplotlib seaborn kagglehub
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ─── Configuration ──────────────────────────────────────────────

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES = 7
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.0005

# Transfer learning uses 96x96 (MobileNetV2 minimum input)
TL_IMG_SIZE = 96

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# STEP 1: LOAD & PREPARE DATASET
# ═══════════════════════════════════════════════════════════════════

def download_fer2013():
    """Download FER-2013 dataset organized as image folders."""
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Check if we have actual content
        count = sum(len(files) for _, _, files in os.walk(train_dir))
        if count > 100:
            print(f"✅ Dataset already exists ({count} training images)")
            return train_dir, test_dir

    print("📥 Downloading FER-2013 dataset...")

    # Try kagglehub first
    try:
        import kagglehub
        path = kagglehub.dataset_download("msambare/fer2013")
        print(f"   Downloaded to: {path}")

        # Find train/test dirs in downloaded path
        for root, dirs, files in os.walk(path):
            if 'train' in dirs and 'test' in dirs:
                src_train = os.path.join(root, 'train')
                src_test = os.path.join(root, 'test')
                # Copy or link
                if not os.path.exists(train_dir):
                    import shutil
                    shutil.copytree(src_train, train_dir)
                if not os.path.exists(test_dir):
                    import shutil
                    shutil.copytree(src_test, test_dir)
                print(f"✅ Dataset organized in: {DATA_DIR}")
                return train_dir, test_dir

        # If no train/test dirs found, check for CSV
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    csv_path = os.path.join(root, f)
                    print(f"   Found CSV: {csv_path}")
                    _csv_to_images(csv_path, train_dir, test_dir)
                    return train_dir, test_dir

    except ImportError:
        print("⚠️  kagglehub not installed.")
        print("   Install with: pip install kagglehub")

    # Check for local CSV
    csv_path = os.path.join(DATA_DIR, "fer2013.csv")
    if os.path.exists(csv_path):
        print(f"📂 Found local CSV: {csv_path}")
        _csv_to_images(csv_path, train_dir, test_dir)
        return train_dir, test_dir

    print("\n" + "=" * 60)
    print("❌ DATASET NOT FOUND")
    print("=" * 60)
    print("Please download FER-2013 in one of these ways:")
    print()
    print("Option 1 (Recommended): pip install kagglehub")
    print("   Then run this script again.")
    print()
    print("Option 2: Manual download")
    print("   1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("   2. Download and extract to:")
    print(f"      {DATA_DIR}/train/  (with subfolders: angry, happy, sad, etc.)")
    print(f"      {DATA_DIR}/test/   (with subfolders: angry, happy, sad, etc.)")
    print()
    print("Option 3: CSV format")
    print(f"   Place fer2013.csv in: {DATA_DIR}")
    sys.exit(1)


def _csv_to_images(csv_path, train_dir, test_dir):
    """Convert FER-2013 CSV to organized image folders."""
    import pandas as pd
    import cv2

    print("🔄 Converting CSV to image folders...")
    df = pd.read_csv(csv_path)

    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    for split_name, split_dir in [('Training', train_dir), ('PublicTest', test_dir), ('PrivateTest', test_dir)]:
        if 'Usage' in df.columns:
            subset = df[df['Usage'] == split_name]
        else:
            # No Usage column, do 80/20 split
            from sklearn.model_selection import train_test_split
            if split_name == 'Training':
                subset, _ = train_test_split(df, test_size=0.2, random_state=42)
            else:
                _, subset = train_test_split(df, test_size=0.2, random_state=42)

        for _, row in subset.iterrows():
            emotion_idx = int(row['emotion'])
            pixels = np.fromstring(row['pixels'], sep=' ').reshape(48, 48).astype(np.uint8)

            emotion_dir = os.path.join(split_dir, emotion_names[emotion_idx])
            os.makedirs(emotion_dir, exist_ok=True)

            img_count = len(os.listdir(emotion_dir))
            img_path = os.path.join(emotion_dir, f"{img_count:05d}.jpg")
            cv2.imwrite(img_path, pixels)

    total_train = sum(len(files) for _, _, files in os.walk(train_dir))
    total_test = sum(len(files) for _, _, files in os.walk(test_dir))
    print(f"   ✓ Train: {total_train} images")
    print(f"   ✓ Test: {total_test} images")


def visualize_dataset(train_dir):
    """Save a visualization of sample images from each emotion class."""
    print("\n🖼️  Visualizing Dataset Samples...")
    import cv2

    emotion_dirs = sorted(os.listdir(train_dir))
    fig, axes = plt.subplots(len(emotion_dirs), 8, figsize=(20, len(emotion_dirs) * 2.5))
    fig.suptitle('FER-2013 Dataset Samples (per Emotion Class)', fontsize=16, fontweight='bold', y=1.01)

    for i, emotion_name in enumerate(emotion_dirs):
        emo_dir = os.path.join(train_dir, emotion_name)
        if not os.path.isdir(emo_dir):
            continue
        images = os.listdir(emo_dir)[:8]
        for j, img_name in enumerate(images):
            img = cv2.imread(os.path.join(emo_dir, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f'{emotion_name.upper()}\n({len(os.listdir(emo_dir))} imgs)',
                                     fontsize=9, fontweight='bold', loc='left')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "dataset_samples.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path}")

    # Also save class distribution
    print("   Generating class distribution chart...")
    classes = []
    counts = []
    for emo in sorted(os.listdir(train_dir)):
        emo_path = os.path.join(train_dir, emo)
        if os.path.isdir(emo_path):
            classes.append(emo.capitalize())
            counts.append(len(os.listdir(emo_path)))

    plt.figure(figsize=(10, 5))
    colors = ['#ef4444', '#ec4899', '#8b5cf6', '#10b981', '#3b82f6', '#f59e0b', '#64748b']
    bars = plt.barh(classes, counts, color=colors[:len(classes)])
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontweight='bold')
    plt.xlabel('Number of Images')
    plt.title('FER-2013 Training Set Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    dist_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"   Saved: {dist_path}")


def create_data_generators(train_dir, test_dir, img_size, batch_size):
    """
    Create training and validation data generators with augmentation.

    Data Augmentation (Training only):
      - Random rotation (±20 degrees)
      - Width/Height shift (±15%)
      - Horizontal flip
      - Zoom (±15%)
      - Shear transformation (±15%)
      - Brightness adjustment

    These augmentations help the model generalize better by creating
    variations of the training images, reducing overfitting.
    """
    print(f"\n🎨 Setting up Data Generators (image size: {img_size}x{img_size})...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.1  # Use 10% of training data for validation
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    color_mode = 'grayscale' if img_size == 48 else 'rgb'

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        shuffle=False
    )

    print(f"   Training samples:   {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples:       {test_gen.samples}")
    print(f"   Classes: {list(train_gen.class_indices.keys())}")

    return train_gen, val_gen, test_gen


# ═══════════════════════════════════════════════════════════════════
# STEP 2: BUILD MODELS
# ═══════════════════════════════════════════════════════════════════

def build_custom_cnn():
    """
    Custom CNN Architecture for Emotion Detection (trained from scratch).

    Architecture:
    ┌──────────────────────────────────────────────────────┐
    │ Input: 48 x 48 x 1 (grayscale face image)           │
    ├──────────────────────────────────────────────────────┤
    │ BLOCK 1: Conv2D(64,3x3) → BN → ReLU                │
    │          Conv2D(64,3x3) → BN → ReLU → MaxPool → DO │
    ├──────────────────────────────────────────────────────┤
    │ BLOCK 2: Conv2D(128,3x3) → BN → ReLU               │
    │          Conv2D(128,3x3) → BN → ReLU → MaxPool → DO│
    ├──────────────────────────────────────────────────────┤
    │ BLOCK 3: Conv2D(256,3x3) → BN → ReLU               │
    │          Conv2D(256,3x3) → BN → ReLU → MaxPool → DO│
    ├──────────────────────────────────────────────────────┤
    │ BLOCK 4: Conv2D(512,3x3) → BN → ReLU               │
    │          Conv2D(512,3x3) → BN → ReLU → MaxPool → DO│
    ├──────────────────────────────────────────────────────┤
    │ GlobalAveragePooling2D                               │
    │ Dense(512) → BN → ReLU → Dropout(0.5)               │
    │ Dense(256) → BN → ReLU → Dropout(0.5)               │
    │ Dense(7, softmax) → Output                           │
    └──────────────────────────────────────────────────────┘
    """
    print("\n🏗️  Building Custom CNN Architecture...")

    model = models.Sequential(name="EmotionDetector_CustomCNN")

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                           input_shape=(IMG_SIZE, IMG_SIZE, 1), name='conv1a'))
    model.add(layers.BatchNormalization(name='bn1a'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1b'))
    model.add(layers.BatchNormalization(name='bn1b'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(layers.Dropout(0.25, name='drop1'))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2a'))
    model.add(layers.BatchNormalization(name='bn2a'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2b'))
    model.add(layers.BatchNormalization(name='bn2b'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(layers.Dropout(0.25, name='drop2'))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3a'))
    model.add(layers.BatchNormalization(name='bn3a'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3b'))
    model.add(layers.BatchNormalization(name='bn3b'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(layers.Dropout(0.25, name='drop3'))

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4a'))
    model.add(layers.BatchNormalization(name='bn4a'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4b'))
    model.add(layers.BatchNormalization(name='bn4b'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool4'))
    model.add(layers.Dropout(0.25, name='drop4'))

    # Classifier
    model.add(layers.GlobalAveragePooling2D(name='gap'))
    model.add(layers.Dense(512, activation='relu', name='fc1'))
    model.add(layers.BatchNormalization(name='bn_fc1'))
    model.add(layers.Dropout(0.5, name='drop_fc1'))
    model.add(layers.Dense(256, activation='relu', name='fc2'))
    model.add(layers.BatchNormalization(name='bn_fc2'))
    model.add(layers.Dropout(0.5, name='drop_fc2'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='output'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("📊 Model Summary:")
    model.summary()
    return model


def build_transfer_learning_model():
    """
    Transfer Learning Model using MobileNetV2.

    MobileNetV2:
      - Pretrained on ImageNet (1.4 million images, 1000 classes)
      - Lightweight architecture optimized for mobile/edge devices
      - Uses Depthwise Separable Convolutions for efficiency
      - We freeze the base layers and train only the top classifier

    Strategy:
      Phase 1: Train only top layers (base frozen) → Learn emotion features
      Phase 2: Fine-tune top 30 layers of base → Adapt low-level features

    Architecture:
    ┌──────────────────────────────────────────────────────┐
    │ Input: 96 x 96 x 3 (RGB, upscaled from grayscale)  │
    ├──────────────────────────────────────────────────────┤
    │ MobileNetV2 Base (pretrained, frozen)               │
    │   - 53 layers, 2.2M parameters                      │
    │   - Extracts rich visual features                   │
    ├──────────────────────────────────────────────────────┤
    │ GlobalAveragePooling2D                               │
    │ Dense(512) → BN → ReLU → Dropout(0.5)               │
    │ Dense(256) → BN → ReLU → Dropout(0.3)               │
    │ Dense(7, softmax) → Output                           │
    └──────────────────────────────────────────────────────┘
    """
    print("\n🏗️  Building Transfer Learning Model (MobileNetV2)...")

    # Load MobileNetV2 pretrained on ImageNet, without top classifier
    base_model = keras.applications.MobileNetV2(
        input_shape=(TL_IMG_SIZE, TL_IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers (use as feature extractor)
    base_model.trainable = False

    # Build complete model
    model = models.Sequential(name="EmotionDetector_MobileNetV2")
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D(name='gap'))
    model.add(layers.Dense(512, activation='relu', name='fc1'))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.Dropout(0.5, name='drop1'))
    model.add(layers.Dense(256, activation='relu', name='fc2'))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.Dropout(0.3, name='drop2'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='output'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"   Base model layers: {len(base_model.layers)}")
    print(f"   Base model params: {base_model.count_params():,} (frozen)")
    print("📊 Model Summary:")
    model.summary()
    return model, base_model


# ═══════════════════════════════════════════════════════════════════
# STEP 3: TRAINING
# ═══════════════════════════════════════════════════════════════════

class LiveTrainingPlot(callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric in self.history.keys():
            if metric in logs:
                self.history[metric].append(logs[metric])
        
        # Plot and save
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history['accuracy'], label='Train', linewidth=2, color='#3b82f6')
        if 'val_accuracy' in logs:
            axes[0].plot(self.history['val_accuracy'], label='Validation', linewidth=2, color='#ef4444')
        axes[0].set_title(f'Live Accuracy — {self.model_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['loss'], label='Train', linewidth=2, color='#3b82f6')
        if 'val_loss' in logs:
            axes[1].plot(self.history['val_loss'], label='Validation', linewidth=2, color='#ef4444')
        axes[1].set_title(f'Live Loss — {self.model_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        hist_path = os.path.join(RESULTS_DIR, f"live_history_{self.model_name}.png")
        plt.savefig(hist_path, dpi=150)
        plt.close(fig)

def train_model(model, train_gen, val_gen, model_name, epochs=EPOCHS):
    """
    Train model with advanced callbacks.

    Callbacks:
      - EarlyStopping: Stop if val_accuracy doesn't improve for 12 epochs
      - ReduceLROnPlateau: Halve LR when val_loss plateaus for 5 epochs
      - ModelCheckpoint: Save best model based on val_accuracy
    """
    print(f"\n🚀 Training {model_name}... (max {epochs} epochs)")
    print("=" * 60)

    best_path = os.path.join(MODEL_DIR, f"{model_name}_best.keras")

    cb = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=12,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            best_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        LiveTrainingPlot(model_name)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=cb,
        verbose=1
    )

    print(f"\n✅ {model_name} Training Complete!")
    return history


def fine_tune_model(model, base_model, train_gen, val_gen, fine_tune_epochs=30):
    """
    Phase 2: Fine-tune the top layers of MobileNetV2.
    Unfreeze the last 30 layers and continue training with a very low LR.
    """
    print("\n🔧 Fine-Tuning MobileNetV2 (unfreezing top 30 layers)...")

    base_model.trainable = True
    # Freeze all except last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Trainable base layers: {trainable_count} / {len(base_model.layers)}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Very low LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = train_model(model, train_gen, val_gen, "fine_tuned", epochs=fine_tune_epochs)
    return history


# ═══════════════════════════════════════════════════════════════════
# STEP 4: EVALUATION & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_and_visualize(model, test_gen, history, model_name):
    """
    Comprehensive evaluation:
      1. Test accuracy & loss
      2. Classification report (precision, recall, F1)
      3. Confusion matrix
      4. Training curves (accuracy & loss)
      5. Sample predictions grid
    """
    print(f"\n📈 Evaluating {model_name}...")
    print("=" * 60)

    # 1. Test metrics
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"\n   ★ Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   ★ Test Loss:     {test_loss:.4f}")

    # 2. Predictions
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())
    class_names_cap = [c.capitalize() for c in class_names]

    # 3. Classification Report
    print(f"\n{'='*60}")
    print("📋 Classification Report:")
    print('=' * 60)
    report = classification_report(y_true, y_pred, target_names=class_names_cap, digits=4)
    print(report)

    report_path = os.path.join(RESULTS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Emotion Detection - {model_name} - Classification Report\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n\n")
        f.write(report)
    print(f"   Saved: {report_path}")

    # 4. Confusion Matrix
    print("\n🔢 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_cap,
                yticklabels=class_names_cap)
    plt.title(f'Confusion Matrix — {model_name}\nAccuracy: {test_accuracy*100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"   Saved: {cm_path}")

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=class_names_cap,
                yticklabels=class_names_cap)
    plt.title(f'Normalized Confusion Matrix — {model_name}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_norm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_normalized_{model_name}.png")
    plt.savefig(cm_norm_path, dpi=150)
    plt.close()
    print(f"   Saved: {cm_norm_path}")

    # 5. Training History
    if history:
        print("\n📊 Generating Training Curves...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, color='#3b82f6')
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#ef4444')
        axes[0].set_title(f'Accuracy — {model_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.history['loss'], label='Train', linewidth=2, color='#3b82f6')
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#ef4444')
        axes[1].set_title(f'Loss — {model_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        hist_path = os.path.join(RESULTS_DIR, f"training_history_{model_name}.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"   Saved: {hist_path}")

    # 6. Sample Predictions
    print("\n🖼️  Generating Sample Predictions...")
    test_gen.reset()
    sample_images, sample_labels = next(test_gen)
    sample_preds = model.predict(sample_images[:18], verbose=0)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for i in range(min(18, len(sample_images))):
        row, col = i // 6, i % 6
        img = sample_images[i]
        if img.shape[-1] == 1:
            axes[row, col].imshow(img.squeeze(), cmap='gray')
        else:
            axes[row, col].imshow(img)
        true_idx = np.argmax(sample_labels[i])
        pred_idx = np.argmax(sample_preds[i])
        true_name = class_names_cap[true_idx]
        pred_name = class_names_cap[pred_idx]
        conf = sample_preds[i][pred_idx] * 100
        color = 'green' if true_idx == pred_idx else 'red'
        axes[row, col].set_title(f'T:{true_name}\nP:{pred_name} ({conf:.0f}%)',
                                  fontsize=9, color=color, fontweight='bold')
        axes[row, col].axis('off')

    plt.suptitle(f'Sample Predictions — {model_name} (Green=Correct, Red=Wrong)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    samples_path = os.path.join(RESULTS_DIR, f"sample_predictions_{model_name}.png")
    plt.savefig(samples_path, dpi=150)
    plt.close()
    print(f"   Saved: {samples_path}")

    return test_accuracy, test_loss


def save_final_model(model, accuracy, model_name):
    """Save the trained model in multiple formats."""
    print(f"\n💾 Saving {model_name}...")

    keras_path = os.path.join(MODEL_DIR, "emotion_model.keras")
    model.save(keras_path)
    print(f"   ✓ {keras_path}")

    h5_path = os.path.join(MODEL_DIR, "emotion_model.h5")
    model.save(h5_path)
    print(f"   ✓ {h5_path}")

    info_path = os.path.join(MODEL_DIR, "model_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 55 + "\n")
        f.write("Emotion Detection Model — Training Report\n")
        f.write("=" * 55 + "\n")
        f.write(f"Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model:         {model_name}\n")
        f.write(f"Dataset:       FER-2013 (35,887 images)\n")
        f.write(f"Classes:       {NUM_CLASSES} ({', '.join(EMOTION_LABELS)})\n")
        f.write(f"Parameters:    {model.count_params():,}\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Batch Size:    {BATCH_SIZE}\n")
        f.write(f"Max Epochs:    {EPOCHS}\n")
        f.write(f"\nFiles saved:\n")
        f.write(f"  emotion_model.keras\n")
        f.write(f"  emotion_model.h5\n")
    print(f"   ✓ {info_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🧠 EMOTION DETECTION — TRAINING PIPELINE")
    print("   Course: Computer Vision & Robotics Intelligence")
    print("   Dataset: FER-2013 (35,887 images, 7 emotions)")
    print("   Models: Custom CNN + Transfer Learning (MobileNetV2)")
    print("=" * 60)

    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 GPU: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠️  No GPU. Training on CPU (will be slower).")

    # Step 1: Download/prepare dataset
    train_dir, test_dir = download_fer2013()
    visualize_dataset(train_dir)

    # ──── Train Custom CNN (48x48 grayscale) ────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: CUSTOM CNN (from scratch)")
    print("=" * 60)

    train_gen_cnn, val_gen_cnn, test_gen_cnn = create_data_generators(
        train_dir, test_dir, IMG_SIZE, BATCH_SIZE
    )
    cnn_model = build_custom_cnn()
    cnn_history = train_model(cnn_model, train_gen_cnn, val_gen_cnn, "custom_cnn", epochs=EPOCHS)
    cnn_acc, cnn_loss = evaluate_and_visualize(cnn_model, test_gen_cnn, cnn_history, "Custom_CNN")

    # ──── Train Transfer Learning (96x96 RGB) ──────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: TRANSFER LEARNING (MobileNetV2 + Fine-tune)")
    print("=" * 60)

    train_gen_tl, val_gen_tl, test_gen_tl = create_data_generators(
        train_dir, test_dir, TL_IMG_SIZE, BATCH_SIZE
    )
    tl_model, base_model = build_transfer_learning_model()

    # Phase 2a: Train top layers only
    tl_history = train_model(tl_model, train_gen_tl, val_gen_tl, "mobilenetv2", epochs=30)
    tl_acc1, _ = evaluate_and_visualize(tl_model, test_gen_tl, tl_history, "MobileNetV2_Frozen")

    # Phase 2b: Fine-tune
    ft_history = fine_tune_model(tl_model, base_model, train_gen_tl, val_gen_tl, fine_tune_epochs=40)
    tl_acc2, tl_loss2 = evaluate_and_visualize(tl_model, test_gen_tl, ft_history, "MobileNetV2_FineTuned")

    # ──── Compare & Save Best ──────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"   Custom CNN Accuracy:             {cnn_acc*100:.2f}%")
    print(f"   MobileNetV2 (frozen) Accuracy:   {tl_acc1*100:.2f}%")
    print(f"   MobileNetV2 (fine-tuned) Accuracy: {tl_acc2*100:.2f}%")

    # Save comparison chart
    plt.figure(figsize=(8, 5))
    models_names = ['Custom CNN', 'MobileNetV2\n(Frozen)', 'MobileNetV2\n(Fine-tuned)']
    accs = [cnn_acc * 100, tl_acc1 * 100, tl_acc2 * 100]
    colors = ['#3b82f6', '#8b5cf6', '#10b981']
    bars = plt.bar(models_names, accs, color=colors, width=0.5)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', fontweight='bold', fontsize=12)
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Comparison — Emotion Detection', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"\n   Comparison chart saved: {comp_path}")

    # Save the best model
    best_acc = max(cnn_acc, tl_acc2)
    if tl_acc2 >= cnn_acc:
        save_final_model(tl_model, tl_acc2, "MobileNetV2_FineTuned")
        print(f"\n🏆 Best model: MobileNetV2 Fine-tuned ({tl_acc2*100:.2f}%)")
    else:
        save_final_model(cnn_model, cnn_acc, "Custom_CNN")
        print(f"\n🏆 Best model: Custom CNN ({cnn_acc*100:.2f}%)")

    print("\n" + "=" * 60)
    print("🎉 ALL TRAINING COMPLETE!")
    print(f"   Models saved to:  {MODEL_DIR}")
    print(f"   Results saved to: {RESULTS_DIR}")
    print(f"   Dataset stored in: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
