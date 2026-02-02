"""
CNN Model Training for Knee Osteoarthritis Severity Assessment
HealthPredict AI by MedIndia

This script trains a CNN model to classify knee X-rays into 5 severity grades:
- Grade 0: Normal (No OA)
- Grade 1: Doubtful (Minimal OA)
- Grade 2: Mild OA
- Grade 3: Moderate OA
- Grade 4: Severe OA

Based on Kellgren-Lawrence (KL) grading system
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
import matplotlib.pyplot as plt
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Knee-Osteoarthritis-Dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
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
    """Create training and validation data generators with augmentation."""

    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data - only rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Test data - only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

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

    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def build_custom_cnn():
    """Build a custom CNN architecture for knee OA classification."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def build_vgg16_transfer():
    """Build VGG16 transfer learning model."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last few layers for fine-tuning
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def build_mobilenet_transfer():
    """Build MobileNetV2 transfer learning model - lightweight and fast."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze most layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def train_model(model, train_gen, val_gen, model_name='knee_cnn'):
    """Train the model with callbacks."""

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Calculate class weights for imbalanced data
    total_samples = sum([2286, 1046, 1516, 757, 173])  # From dataset
    class_weights = {
        0: total_samples / (5 * 2286),
        1: total_samples / (5 * 1046),
        2: total_samples / (5 * 1516),
        3: total_samples / (5 * 757),
        4: total_samples / (5 * 173)
    }

    print(f"\nClass weights: {class_weights}")

    # Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return history


def evaluate_model(model, test_gen):
    """Evaluate model on test set."""
    results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")
    return results


def save_model_for_serving(model, model_name='knee_oa_cnn'):
    """Save model for Flask API serving."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save full model
    model_path = os.path.join(MODELS_DIR, f'{model_name}.keras')
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save model info
    info = {
        'model_name': model_name,
        'input_shape': (224, 224, 3),
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'severity_descriptions': SEVERITY_DESCRIPTIONS
    }

    info_path = os.path.join(MODELS_DIR, f'{model_name}_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved to: {info_path}")

    return model_path


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to: {save_path}")

    plt.show()


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Knee Osteoarthritis CNN Training")
    print("HealthPredict AI by MedIndia")
    print("=" * 60)

    # Check GPU
    print(f"\nTensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")

    # Create data generators
    print("\nLoading dataset...")
    train_gen, val_gen, test_gen = create_data_generators()

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {train_gen.class_indices}")

    # Build model - using MobileNetV2 for balance of speed and accuracy
    print("\nBuilding MobileNetV2 transfer learning model...")
    model = build_mobilenet_transfer()
    model.summary()

    # Train
    print("\nStarting training...")
    history = train_model(model, train_gen, val_gen, 'knee_oa_mobilenet')

    # Evaluate
    print("\nEvaluating on test set...")
    evaluate_model(model, test_gen)

    # Save model
    print("\nSaving model...")
    save_model_for_serving(model, 'knee_oa_cnn')

    # Plot history
    plot_training_history(history, os.path.join(MODELS_DIR, 'knee_oa_training_history.png'))

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
