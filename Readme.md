## Alphabet Gestures

The `alphabet gestures` folder contains scripts for processing and training models on sign language gestures corresponding to each letter of the alphabet. The structure is as follows:

### Files in `alphabet gestures`:

- **collect_imgs.py**: Captures images from a webcam and saves them to the dataset.
- **create_dataset.py**: Processes images to create datasets for training.
- **image_augmentation.py**: Applies various image augmentation techniques.
- **inference_classifier.py**: Loads trained models and performs inference on new data.
- **train_classifier.py**: Trains classifiers on the processed datasets.

## Word Gestures

The `word gestures` folder contains scripts for processing and training models on sign language gestures corresponding to different words. The structure is as follows:

### Files in `word gestures`:

- **extract_landmarks.py**: Extracts landmarks from videos.
- **gesture_recognition_model.h5**: Pre-trained model for gesture recognition.
- **load_landmarks.py**: Loads and displays the content of a `.npy` file containing landmarks.
- **predict_gesture.py**: Loads a trained model and uses it to predict the gesture in a test video.
- **preprocessing_testing_videos/**: Folder containing scripts for preprocessing testing videos.
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
- **scikit-learn**: Used for encoding labels and splitting data for training and testing.

---
