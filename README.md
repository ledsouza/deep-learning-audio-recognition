## Audio Recognition System for Security and Accessibility 🔐 

![Static Badge](https://img.shields.io/badge/Status-Finalized-green)

## Description

This project aims to build two Machine Learning models for audio recognition, focusing on security and accessibility. The first model, built from scratch, recognizes audio commands for accessibility, using Short-Time Fourier Transform (STFT) to transform audios into images that represent the signal in time and frequency. The second model, based on Transfer Learning, utilizes the YAMNet pre-trained model to recognize sounds related to security threats, such as breaking glass.

## Technologies Used

- Python
- TensorFlow
- TensorFlow Hub
- Librosa
- SciPy
- NumPy
- Pandas
- Matplotlib

## Detailed Project Description

### Model 1: Audio Command Recognition for Accessibility

- **Objective:** Recognize specific voice commands to enhance accessibility.
- **Data:** Dataset of audio commands.
- **Preprocessing:**
    - Short-Time Fourier Transform (STFT) to convert audio into a spectrogram (time-frequency representation).
- **Model Architecture:**
    - Convolutional Neural Network (CNN) with convolutional, pooling, and fully connected layers.
    - Utilization of Channel Attention to improve the model's ability to focus on relevant features.
- **Training:** The model was trained from scratch using the audio command data.

### Model 2: Audio Recognition for Security

- **Objective:** Identify sounds that may indicate security threats, such as the sound of breaking glass.
- **Data:** ESC-50 dataset, filtered to include only classes relevant to security (e.g., dog barking, door wood creaks, glass breaking).
- **Preprocessing:**
    - Loading MP3 audio files.
    - Resampling to 16kHz if necessary.
    - Padding or slicing the signal to a fixed length.
- **Transfer Learning:**
    - Using the YAMNet pre-trained model to extract audio embeddings.
    - Training a specialized model with the extracted embeddings, focusing on the security classes.
- **Model Architecture:**
    - Neural network with dense (fully connected) layers.
    - The output layer has 3 neurons, corresponding to the 3 security classes.
- **Training:** The model was trained with the embeddings extracted from YAMNet, using 5-fold cross-validation.

## Results

- The audio command recognition model for accessibility showed good accuracy during training.
- The audio recognition model for security, using Transfer Learning, achieved high accuracy in validation, demonstrating the effectiveness of the approach.

## Conclusions

This project demonstrates the feasibility of building audio recognition systems for different applications, such as security and accessibility. Using Transfer Learning with pre-trained models, like YAMNet, allows building specialized models with high accuracy, even with smaller datasets.
