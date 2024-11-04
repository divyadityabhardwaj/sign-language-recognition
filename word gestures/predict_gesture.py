import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:/study/sign language trial/gesture_recognition_model.h5')

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

def normalize_landmarks(landmarks):
    if landmarks is None:
        return None
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    # Normalize to the range [0, 1]
    return (landmarks - np.min(landmarks, axis=0)) / (np.max(landmarks, axis=0) - np.min(landmarks, axis=0))

def pad_landmarks(landmarks_list, target_length=180):
    num_frames = len(landmarks_list)
    if num_frames < target_length:
        padding = np.zeros((target_length - num_frames, 225))  # Shape of padding for flattened landmarks
        return np.concatenate((landmarks_list, padding), axis=0)
    elif num_frames > target_length:
        return landmarks_list[:target_length]
    return landmarks_list

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Extract and normalize pose landmarks
        pose_landmarks = normalize_landmarks(results.pose_landmarks)
        left_hand_landmarks = normalize_landmarks(results.left_hand_landmarks)
        right_hand_landmarks = normalize_landmarks(results.right_hand_landmarks)

        # Combine all landmarks into a single array
        if pose_landmarks is not None:
            combined_landmarks = np.concatenate((
                pose_landmarks,
                left_hand_landmarks if left_hand_landmarks is not None else np.zeros((21, 3)),
                right_hand_landmarks if right_hand_landmarks is not None else np.zeros((21, 3))
            ), axis=0)

            # Flatten combined landmarks to match expected input shape
            flattened_landmarks = combined_landmarks.flatten()  # Shape: (225,)
            landmarks_list.append(flattened_landmarks)

    cap.release()

    # Convert landmarks list to numpy array and ensure it's 2D
    if len(landmarks_list) == 0:
        return np.zeros((180, 225))  # Return zeros if no frames were processed

    return np.array(landmarks_list)

# Path to the input test video
video_path = ""

# Extract landmarks from the test video
landmarks_array = extract_landmarks_from_video(video_path)

# Pad the extracted landmarks to ensure consistent input shape for prediction
padded_landmarks = pad_landmarks(landmarks_array, target_length=180)

# Reshape for model input: (1, target_length, features)
input_data = padded_landmarks.reshape(1, padded_landmarks.shape[0], 225)  # Add batch dimension

# Make predictions using the model
predictions = model.predict(input_data)
predicted_class = np.argmax(predictions)
confidence = predictions[0][predicted_class]

# Output the predicted class and confidence
print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")