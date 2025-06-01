import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

print("Starting script...")

# Use raw strings or forward slashes for Windows paths
train_dir = r"C:\Users\muhammed sadiq\Desktop\pneumonia_detection_cnn\chest_xray\train"
val_dir = r"C:\Users\muhammed sadiq\Desktop\pneumonia_detection_cnn\chest_xray\val"

print("Checking if directories exist:")
print("Train dir exists?", os.path.exists(train_dir))
print("Val dir exists?", os.path.exists(val_dir))

img_height, img_width = 150, 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

print("Creating data generators...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

print("Train generator classes:", train_generator.class_indices)
print("Validation generator classes:", val_generator.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Starting training...")
history = model.fit(
    train_generator,
    epochs=6,
    validation_data=val_generator
)

print("Training completed, saving model...")
model.save(r"C:\Users\muhammed sadiq\Desktop\pneumonia_detection_cnn\pneumonia_cnn_model.h5")
print("Model saved!")
