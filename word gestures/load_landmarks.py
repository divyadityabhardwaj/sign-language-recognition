import numpy as np

# Path to the npy file
npy_file_path = r"D:\study\sign language trial\\new\\01610.npy"  # Update with your actual path

# Load landmarks from npy file
landmarks_array = np.load(npy_file_path)

# Display the contents of the npy file
print("Contents of the npy file:")
print(landmarks_array)

# Optionally, display the shape of the array
print("\nShape of the landmarks array:", landmarks_array.shape)

# Check if landmarks are normalized
is_normalized = np.all((landmarks_array[:, :, 0] >= 0) & (landmarks_array[:, :, 0] <= 1)) and \
                np.all((landmarks_array[:, :, 1] >= 0) & (landmarks_array[:, :, 1] <= 1))

print("Are the landmarks normalized? ", is_normalized)