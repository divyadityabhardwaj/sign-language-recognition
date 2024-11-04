#landmark npy file without normaliszation can be used to output videos , but the normalized ones , which are better to train a model on do not give a good output after normalization 
import cv2
import numpy as np

# Path to the npy file containing normalized landmarks
npy_file_path = r"new\\01610.npy"  # Update with your actual path
output_video_path = r"D:\study\sign language trial\output3.avi"  # Path to save the output video

# Load normalized landmarks from npy file
landmarks_array = np.load(npy_file_path)

# Original frame dimensions
original_frame_width = 640  # Set your original frame width
original_frame_height = 480  # Set your original frame height

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # Set your desired frames per second
out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_frame_width, original_frame_height))

# Function to draw landmarks on a frame
def draw_landmarks(frame, landmarks):
    # Assuming the first 33 landmarks are for the pose and the next 42 are for the hands
    pose_landmarks = landmarks[:33]  # Adjust based on your landmark structure
    left_hand_landmarks = landmarks[33:54]  # Adjust based on your landmark structure
    right_hand_landmarks = landmarks[54:75]  # Adjust based on your landmark structure

    # Draw pose landmarks in green
    for lm in pose_landmarks:
        x, y = int(lm[0] * original_frame_width), int(lm[1] * original_frame_height)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green for pose landmarks

    # Draw left hand landmarks in red
    for lm in left_hand_landmarks:
        x, y = int(lm[0] * original_frame_width), int(lm[1] * original_frame_height)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for left hand landmarks

    # Draw right hand landmarks in red
    for lm in right_hand_landmarks:
        x, y = int(lm[0] * original_frame_width), int(lm[1] * original_frame_height)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for right hand landmarks
# Iterate through the landmarks and create frames
for landmarks in landmarks_array:
    # Create a blank frame
    frame = np.zeros((original_frame_height, original_frame_width, 3), dtype=np.uint8)

    # Draw landmarks on the frame
    draw_landmarks(frame, landmarks)

    # Write the frame to the output video
    out.write(frame)

# Release resources
out.release()
cv2.destroyAllWindows()

print("Landmark visualization completed. Output saved to:", output_video_path)
