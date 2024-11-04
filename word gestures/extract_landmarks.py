import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Path to the base folder containing label folders
base_folder = r"D:\study\sign language trial\sl_test"
# Create a parent folder for landmark npy files
landmark_folder = os.path.join(base_folder, "landmarks_npy")
os.makedirs(landmark_folder, exist_ok=True)

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    if landmarks is None:
        return None
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    # Normalize to the range [0, 1]
    return (landmarks - np.min(landmarks, axis=0)) / (np.max(landmarks, axis=0) - np.min(landmarks, axis=0))

# Initialize counters
total_videos = 0
total_frames = 0

# Iterate through each label folder
for label in os.listdir(base_folder):
    label_path = os.path.join(base_folder, label)
    if os.path.isdir(label_path):
        # Create a folder for the label in the landmarks_npy directory
        label_landmark_folder = os.path.join(landmark_folder, label)
        os.makedirs(label_landmark_folder, exist_ok=True)

        # Iterate through the video files in the label folder
        for filename in os.listdir(label_path):
            if filename.endswith(".mp4") or filename.endswith(".avi"):  # Add other video formats if needed
                video_path = os.path.join(label_path, filename)
                cap = cv2.VideoCapture(video_path)

                landmarks_list = []  # List to store landmarks for the current video
                frame_count = 0  # Counter for frames in the current video

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
                        combined_landmarks = np.concatenate((pose_landmarks, 
                                                             left_hand_landmarks if left_hand_landmarks is not None else np.zeros((21, 3)),
                                                             right_hand_landmarks if right_hand_landmarks is not None else np.zeros((21, 3))), axis=0)
                        landmarks_list.append(combined_landmarks)
                        frame_count += 1  # Increment frame counter

                cap.release()

                # Convert landmarks list to numpy array and save
                landmarks_array = np.array(landmarks_list)
                npy_filename = os.path.splitext(filename)[0] + ".npy"
                np.save(os.path.join(label_landmark_folder, npy_filename), landmarks_array)

                # Update total counters
                total_videos += 1
                total_frames += frame_count

                print(f"Processed {filename}: {frame_count} frames of landmarks saved.")

# Print final counts
print(f"\nTotal Videos Processed: {total_videos}")
print(f"Total Frames Extracted: {total_frames}")
print("Landmark extraction completed.")