import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import os

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/coffee_model.h5")

# =========================
# PATH VALIDATION
# =========================
dataset_path = "valid"

datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),  # 🔥 UBAH KE 256
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# =========================
# LABEL
# =========================
print("Class Index:", val_data.class_indices)

class_labels = list(val_data.class_indices.keys())

# =========================
# PREDIKSI
# =========================
Y_true = val_data.classes
Y_pred = model.predict(val_data)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title('Confusion Matrix (Validation)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig("model/confusion_matrix.png")
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=class_labels))

# =========================
# GRAFIK TRAINING
# =========================
if os.path.exists("model/history.json"):
    with open("model/history.json", "r") as f:
        history = json.load(f)

    plt.figure()

    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')

    plt.legend()
    plt.title("Training Accuracy & Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.savefig("model/training_graph.png")
    plt.show()
else:
    print("⚠️ History tidak ditemukan. Jalankan training ulang.")