import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def pad_data(data, target_frames=180):
    current_frames = data.shape[0]
    if current_frames < target_frames:
        padding = np.zeros((target_frames - current_frames, data.shape[1], data.shape[2]))
        return np.concatenate((data, padding), axis=0)
    else:
        return data[:target_frames]

def add_noise(data, noise_factor=0.01):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    return data + noise

def shift_time(data, shift_range=5):
    shift = np.random.randint(-shift_range, shift_range)
    if shift > 0:
        return np.concatenate((data[shift:], np.zeros((shift, data.shape[1], data.shape[2]))), axis=0)
    else:
        return np.concatenate((np.zeros((-shift, data.shape[1], data.shape[2])), data[:shift]), axis=0)

def process_data(parent_folder):
    all_data = []
    all_labels = []

    for label in os.listdir(parent_folder):
        label_folder = os.path.join(parent_folder, label)

        if os.path.isdir(label_folder):
            for file in os.listdir(label_folder):
                if file.endswith('.npy'):
                    file_path = os.path.join(label_folder, file)
                    data = np.load(file_path)
                    padded_data = pad_data(data)

                    # Original data
                    all_data.append(padded_data)
                    all_labels.append(label)

                    # Augmented data
                    augmented_data_noise = add_noise(padded_data)
                    augmented_data_shifted = shift_time(padded_data)

                    all_data.append(pad_data(augmented_data_noise))
                    all_labels.append(label)
                    all_data.append(pad_data(augmented_data_shifted))
                    all_labels.append(label)

    return np.array(all_data), np.array(all_labels)

# Load and preprocess the data
parent_folder = 'landmarks_npy'  # Update with your actual path
X, y = process_data(parent_folder)

# Reshape X for LSTM input: (num_samples, time_steps, features)
num_samples = X.shape[0]
time_steps = X.shape[1]
features = X.shape[2] * X.shape[3]  # Flatten the last two dimensions

X_reshaped = X.reshape(num_samples, time_steps, features)

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert labels to one-hot encoding
y_one_hot = keras.utils.to_categorical(y_encoded)

# Split the Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_one_hot, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(time_steps, features)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(y_one_hot.shape[1], activation='softmax')  # Output layer for classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=35, batch_size=32,
                    validation_data=(X_val, y_val))

import matplotlib.pyplot as plt

# Plotting training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Show plots
plt.tight_layout()
plt.show()

# Save the model
model.save('gesture_recognition_model.h5')

# Assuming label_encoder is defined and fitted during training
class_names = list(label_encoder.classes_)  # Get the class names from the encoder

# Print all labels with their corresponding class names
for index, class_name in enumerate(class_names):
    print(f'Label {index}: {class_name}')

