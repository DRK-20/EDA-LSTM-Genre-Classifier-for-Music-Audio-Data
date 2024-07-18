# 🎵 LSTM-based Music Genre Classification 🎶

Welcome to the LSTM-based Music Genre Classification project! This repository contains the code and resources for building a robust music genre classification model using Exploratory Data Analysis (EDA) and Long Short-Term Memory (LSTM) neural networks. Our goal is to accurately classify music tracks into their respective genres based on extracted audio features. 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Real-time Inference](#real-time-inference)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

Music genre classification is essential in the field of music information retrieval and has applications in music recommendation systems, automated playlist generation, and digital music libraries. This project leverages the power of LSTM neural networks to handle the sequential nature of audio data, providing a highly accurate genre classification system.

## Features

- **Audio Preprocessing**: Resampling, normalization, and segmentation of audio tracks.
- **Feature Extraction**: Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio tracks.
- **LSTM Model Architecture**: Robust and scalable model design for sequential data.
- **Evaluation Metrics**: Includes accuracy, precision, recall, and F1-score.
- **Real-time Inference**: Classify genres of music tracks in real-time.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- NumPy
- Pandas
- Librosa
- TensorFlow/Keras
- Matplotlib
- Seaborn
- Scikit-learn

## Usage

### Data Preprocessing

1. **Load and preprocess the data**:
    ```python
    import pandas as pd
    from preprocessing import preprocess_audio, extract_mfcc, segment_audio

    data = pd.read_csv('data/music_genre_dataset.csv')

    # Example preprocessing
    y, sr = preprocess_audio('data/song.wav')
    mfccs = extract_mfcc(y, sr)
    segments = segment_audio(mfccs)
    ```

### Model Training

2. **Build and train the model**:
    ```python
    from model import build_lstm_model

    model = build_lstm_model(input_shape=(None, 13), num_classes=10)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
    ```

### Evaluation

3. **Evaluate the model**:
    ```python
    from evaluation import plot_history

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.2f}')

    plot_history(history)
    ```

### Real-time Inference

4. **Classify genre of a new track**:
    ```python
    from inference import classify_genre

    predicted_genre = classify_genre('data/new_song.wav')
    print(f'Predicted Genre: {predicted_genre}')
    ```

## Results

The LSTM-based model demonstrates high accuracy in classifying music tracks into their respective genres. Detailed evaluation metrics including accuracy, precision, recall, and F1-score indicate the model's robustness and reliability. The plots of training and validation accuracy/loss over epochs provide insights into the model's performance and convergence.

## Acknowledgements

- Special thanks to the open-source community for providing valuable resources and libraries.
- Thanks to all contributors for their hard work and dedication.

---

Feel free to reach out with any questions or feedback! Happy coding and enjoy the music! 🎵🚀

---

## Example of Expected Output

```plaintext
Loading data...
Preprocessing audio...
Extracting MFCCs...
Building model...
Training model...
Epoch 1/50
...
Epoch 50/50
Test Accuracy: 0.92
Predicted Genre: Rock
