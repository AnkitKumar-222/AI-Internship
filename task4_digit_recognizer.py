# ============================================================
# TASK 4: HANDWRITTEN DIGIT RECOGNIZER (MNIST)
# Internship AI Task - Kodbud
# Tools: Python, TensorFlow/Keras, Deep Learning
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 55)
print("  HANDWRITTEN DIGIT RECOGNIZER - AI Internship Task 4")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1: IMPORT TENSORFLOW
# ─────────────────────────────────────────
print("\n[1/6] Importing TensorFlow / Keras...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

print(f"      ✅ TensorFlow version: {tf.__version__}")

# ─────────────────────────────────────────
# STEP 2: LOAD MNIST DATASET
# ─────────────────────────────────────────
print("\n[2/6] Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"      📊 Training samples : {X_train.shape[0]}")
print(f"      📊 Test samples     : {X_test.shape[0]}")
print(f"      📊 Image shape      : {X_train.shape[1]}x{X_train.shape[2]} pixels")

# ─────────────────────────────────────────
# STEP 3: PREPROCESS DATA
# ─────────────────────────────────────────
print("\n[3/6] Preprocessing data...")

# Normalize pixel values from [0,255] → [0,1]
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm  = X_test.astype('float32')  / 255.0

# Flatten 28x28 images to 784 for dense network
X_train_flat = X_train_norm.reshape(-1, 28 * 28)
X_test_flat  = X_test_norm.reshape(-1, 28 * 28)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

print("      ✅ Normalized pixel values to [0,1]")
print("      ✅ Flattened images to 784-dim vector")
print("      ✅ One-hot encoded labels (10 classes)")

# ─────────────────────────────────────────
# STEP 4: BUILD NEURAL NETWORK
# ─────────────────────────────────────────
print("\n[4/6] Building Neural Network architecture...")

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(784,)),

    # Hidden layer 1 - 256 neurons + Batch Norm + Dropout
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Hidden layer 2 - 128 neurons + Batch Norm + Dropout
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Hidden layer 3 - 64 neurons
    layers.Dense(64, activation='relu'),

    # Output layer - 10 classes (digits 0-9)
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────
# STEP 5: TRAIN THE MODEL
# ─────────────────────────────────────────
print("\n[5/6] Training the Neural Network...")

# Callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

history = model.fit(
    X_train_flat, y_train_cat,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# STEP 6: EVALUATE + VISUALIZE
# ─────────────────────────────────────────
print("\n[6/6] Evaluating model on test set...")
test_loss, test_acc = model.evaluate(X_test_flat, y_test_cat, verbose=0)

print(f"\n      🎯 Test Accuracy : {test_acc * 100:.2f}%")
print(f"      📉 Test Loss     : {test_loss:.4f}")

# ─── PLOT 1: Training History ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MNIST Digit Recognizer - Training History', fontsize=14, fontweight='bold')

axes[0].plot(history.history['accuracy'], label='Train Acc', color='royalblue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='darkorange', linewidth=2)
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', color='royalblue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', color='darkorange', linewidth=2)
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task4_training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("      📊 Training history saved as: task4_training_history.png")

# ─── PLOT 2: Sample Predictions ───
print("\n      🔍 Showing sample predictions...")
predictions = model.predict(X_test_flat[:25], verbose=0)

fig, axes = plt.subplots(5, 5, figsize=(12, 12))
fig.suptitle(f'Sample Predictions (Accuracy: {test_acc*100:.2f}%)', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap='gray')
    pred_digit  = np.argmax(predictions[i])
    true_digit  = y_test[i]
    confidence  = np.max(predictions[i]) * 100
    color = 'green' if pred_digit == true_digit else 'red'
    ax.set_title(f'Pred: {pred_digit} ({confidence:.0f}%)\nTrue: {true_digit}',
                 color=color, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('task4_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("      📊 Predictions grid saved as: task4_predictions.png")

# ─── SAVE MODEL ───
model.save('mnist_digit_model.h5')
print("      💾 Model saved as: mnist_digit_model.h5")

# ─────────────────────────────────────────
# BONUS: Predict a single custom digit
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("         🔍 PREDICTING RANDOM TEST IMAGES")
print("=" * 55)

# Pick 10 random test samples
import random
indices = random.sample(range(len(X_test)), 10)

print()
for idx in indices:
    img   = X_test_flat[idx].reshape(1, -1)
    pred  = model.predict(img, verbose=0)
    digit = np.argmax(pred)
    conf  = np.max(pred) * 100
    true  = y_test[idx]
    status = "✅" if digit == true else "❌"
    print(f"  {status} Predicted: {digit}  |  True: {true}  |  Confidence: {conf:.1f}%")

print("\n" + "=" * 55)
print("  ✅ Task 4 Complete! Digit Recognizer is working.")
print(f"  🎯 Final Test Accuracy: {test_acc * 100:.2f}%")
print("=" * 55)
