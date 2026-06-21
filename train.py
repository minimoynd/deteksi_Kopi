import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os

# =========================
# FOLDER MODEL
# =========================
os.makedirs("model", exist_ok=True)

# =========================
# DATASET
# =========================
dataset_path = "dataset"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Class Index:", train_data.class_indices)

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3) 
)

# Freeze sebagian layer
for layer in base_model.layers[:120]:
    layer.trainable = False

for layer in base_model.layers[120:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    
    # Layer Dense pertama
    layers.Dense(256, activation='relu'),
    #  UBAH: Dropout dinaikkan dari 0.5 menjadi 0.6
    layers.Dropout(0.6), 
    
    # Layer Dense kedua
    layers.Dense(128, activation='relu'),
    #  UBAH: Dropout dinaikkan dari 0.3 menjadi 0.5
    layers.Dropout(0.5), 
    
    # Layer Output
    layers.Dense(4, activation='softmax')  #  4 KELAS
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACK
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "model/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# =========================
# CLASS WEIGHT
# =========================
class_weight = {
    0: 2.0,
    1: 1.0,
    2: 1.5,
    3: 2.0
}

# =========================
# TRAINING
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =========================
# SAVE MODEL
# =========================
model.save("model/coffee_model.h5")

# =========================
# SAVE HISTORY
# =========================
with open("model/history.json", "w") as f:
    json.dump(history.history, f)

print("MODEL FINAL SIAP!")