---

## Alphabet Gestures

The `alphabet gestures` folder contains scripts for processing and training models on sign language gestures corresponding to each letter of the alphabet. The structure is as follows:


### Files in `alphabet gestures`:

- **collect_imgs.py**: Captures images from a webcam and saves them to the dataset.
- **create_dataset.py**: Processes images to create datasets for training.
- **image_augmentaion.py**: Applies various image augmentation techniques.
- **inference_classifier.py**: Loads trained models and performs inference on new data.
- **train_classifier.py**: Trains classifiers on the processed datasets.

## Word Gestures

The `word gestures` folder contains scripts for processing and training models on sign language gestures corresponding to different words. The structure is as follows:


### Files in `word gestures`:

- **extract_landmarks.py**: Extracts landmarks from videos.
- **gesture_recognition_model.h5**: Pre-trained model for gesture recognition.
- **load_landmarks.py**: Loads and displays the content of a `.npy` file containing landmarks.
- **predict_gesture.py**: Loads a trained model and uses it to predict the gesture in a test video.
- **preprocessing testing videos/**: Folder containing scripts for preprocessing testing videos.
- **train.py**: Preprocesses landmark data with data augmentation and trains a neural network model.
- **visualize_landmarks.py**: Visualizes normalized landmarks by drawing them on a blank frame and saving it as a video.

# Project Overview

This project focuses on sign language gesture recognition using landmarks extracted from videos. The code processes videos to generate and normalize landmark data, trains a model with data augmentation, and makes predictions using a pre-trained model. Below is an overview of each file and the libraries used.

## Libraries Used

The project leverages several key libraries:
- **NumPy**: Used for array operations and handling `.npy` files containing landmark data.
- **OpenCV**: Handles video processing, capturing frames from video files, and drawing landmarks on frames.
- **MediaPipe**: Provides the `Holistic` solution for extracting pose and hand landmarks from video frames.
- **TensorFlow/Keras**: For building and loading a neural network model to classify sign language gestures.
- **sklearn**: Used for encoding labels and splitting data for training and testing.

---

## File Summaries

### 1. `load_landmarks.py`
- **Purpose**: Loads and displays the content of a `.npy` file containing landmarks.
- **Main Operations**:
  - Loads a landmarks array from a specified `.npy` file path.
  - Checks if the landmarks are normalized (i.e., values between 0 and 1).
  - Prints the shape of the array and whether the data is normalized.
- **Libraries Used**: `numpy`

### 2. `extract_landmarks.py`
- **Purpose**: Extracts and normalizes landmarks from videos, saving them in a structured folder.
- **Main Operations**:
  - Loads videos from label folders and processes each frame to extract pose and hand landmarks using MediaPipe.
  - Normalizes landmarks to the range `[0, 1]`.
  - Saves the landmarks for each video in `.npy` format.
- **Libraries Used**: `opencv-python`, `mediapipe`, `numpy`, `os`
- **Output**: `.npy` files containing normalized landmarks for each video.

### 3. `visualize_landmarks.py`
- **Purpose**: Visualizes normalized landmarks by drawing them on a blank frame and saving it as a video.
- **Main Operations**:
  - Loads normalized landmarks from a `.npy` file.
  - Draws landmarks (pose and hands) on a blank frame.
  - Writes each frame to create a video output, which visualizes the landmark positions.
- **Libraries Used**: `opencv-python`, `numpy`
- **Output**: A video file showing landmarks visualized on frames.

### 4. `predict_gesture.py`
- **Purpose**: Loads a trained model and uses it to predict the gesture in a test video.
- **Main Operations**:
  - Extracts and normalizes landmarks from a test video.
  - Pads the landmarks to ensure consistent input shape for the model.
  - Loads a pre-trained model and predicts the gesture class.
  - Prints the predicted class and confidence level.
- **Libraries Used**: `opencv-python`, `mediapipe`, `tensorflow`, `numpy`
- **Output**: Prediction results with class label and confidence score.

### 5. `train.py`
- **Purpose**: Preprocesses landmark data with data augmentation and trains a neural network model.
- **Main Operations**:
  - Loads `.npy` files from folders, pads each to ensure a consistent frame count.
  - Performs data augmentation by adding noise and shifting the time series data.
  - Uses a simple Keras model architecture for training on the augmented data.
  - Splits data into training and testing sets, encodes labels, and trains the model.
- **Libraries Used**: `numpy`, `tensorflow`, `sklearn`
- **Output**: A trained model that can be saved and used for gesture prediction.

---

## Additional Notes

- **Data Structure**: Landmark `.npy` files are saved in a structured folder hierarchy where each subfolder represents a gesture label.
- **Model Architecture**: The neural network model uses sequential layers with Keras, designed to classify gestures based on landmarks.
- **Normalization**: Normalizing landmarks is essential for consistent model input, but it may affect visualization output (as noted in `visualize_landmarks.py`).
- **Augmentation**: Data augmentation is applied in `train.py` to enhance model generalizability.

## Dataset used for word gestures

https://www.kaggle.com/datasets/amanbind/smallwlasl
---