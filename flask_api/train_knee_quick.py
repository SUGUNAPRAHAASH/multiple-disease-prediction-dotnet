"""
Quick CNN Model Training for Knee OA - For Testing Purposes
Uses fewer epochs for faster training on CPU
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Knee-Osteoarthritis-Dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model parameters - reduced for quick training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Reduced epochs for quick training
NUM_CLASSES = 5

# Class labels
CLASS_NAMES = {
    0: "Normal (Grade 0)",
    1: "Doubtful (Grade 1)",
    2: "Mild (Grade 2)",
    3: "Moderate (Grade 3)",
    4: "Severe (Grade 4)"
}

SEVERITY_DESCRIPTIONS = {
    0: "No signs of osteoarthritis. The knee joint appears healthy with normal cartilage space.",
    1: "Doubtful narrowing of joint space with possible osteophytic lipping. Very early signs that may or may not indicate OA.",
    2: "Definite osteophytes and possible narrowing of joint space. Mild osteoarthritis is present.",
    3: "Moderate multiple osteophytes, definite narrowing of joint space, some sclerosis. Moderate OA requiring attention.",
    4: "Large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity. Severe OA - medical intervention needed."
}


def create_data_generators():
    """Create training and validation data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator


def build_model():
    """Build MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze all base layers for quick training
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def main():
    print("=" * 60)
    print("Quick Knee OA CNN Training (5 epochs)")
    print("=" * 60)

    print(f"\nTensorFlow version: {tf.__version__}")

    # Create data generators
    print("\nLoading dataset...")
    train_gen, val_gen = create_data_generators()
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    # Build model
    print("\nBuilding model...")
    model = build_model()

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Class weights
    class_weights = {
        0: 0.5, 1: 1.1, 2: 0.76, 3: 1.5, 4: 6.7
    }

    # Train
    print("\nTraining for 5 epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        verbose=1
    )

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'knee_oa_cnn.keras')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save model info
    info = {
        'model_name': 'Knee OA CNN (Quick)',
        'input_shape': [224, 224, 3],
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'severity_descriptions': SEVERITY_DESCRIPTIONS,
        'epochs_trained': EPOCHS,
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    }

    info_path = os.path.join(MODELS_DIR, 'knee_oa_cnn_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved to: {info_path}")

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
