# Project Repository 

Welcome to the project repository! This repository contains various projects focusing on Machine Learning, Data Analysis, and more. Each project is aimed at exploring different concepts and techniques within the field of Data Science. 

## Project 1: Stock Price Prediction using LSTM(Long Short-Term Memory)

### Description 

This project aims to predict stock prices using Long Short-Term Memory (LSTM) Neural Networks. The dataset used here is historical stock price data for Google (GOOG). The LSTM model is trained on a portion of the data and tested on the remaining data to predict future stock prices. 

### Requirements 

    Python 3.x 

    Libraries: 

        NumPy 

        Pandas 

        Matplotlib 

        Scikit-learn 

        TensorFlow 

### Usage 

1. Ensure you have the necessary dataset. In this case, make sure to have the “GOOG.csv” file containing the historical Google stock price data. The file has already been added to the Google Prediction project repository.
2. Run the google_stock_price_prediction.py file to train the LSTM model and make predictions.

### Model
The LSTM model used for stock price prediction is defined in the provided Python script. It contains of multiple Bidirectional LSTM layers followed by dropout layers for regularization. The model is trained using the Adam optimizer with a Mean Squared Error loss function.

### Saving the model 
The trained model is saved in a pickle file named "Google_model.pkl". You can load this model later for making predictions without retraining.


## Project 2: Audio Classification

### Description

This project contains code for Deep Audio Classification using TensorFlow and TensorFlow IO libraries. The code includes preprocessing of audio data, building a convolutional neural network (CNN) model, training the model, making predictions on audio clips, and saving the results in a csv file.

### Requirements

-Install Required Libraries
   
    !pip install tensorflow tensorflow-io

-Mount Google Drive
   
    from google.colab import drive
    drive.mount("/content/drive")

-Extrtact Audio Data
   
    with zipfile.ZipFile('/content/drive/MyDrive/Audio_Classification_Data/Data.zip') as z:
        z.extractall()

-Python Dependencies
```py
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from google.colab import drive
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from itertools import groupby
import csv
```
### Usage

1. Data Preprocessing: The code includes functions for loading and preprocessing audio data. It resamples the audio to 16 kHz, extracts spectrograms, and prepares the data for model training.
2. Model Building: A CNN model is constructed using TensorFlow's Sequential API. The model consists of convolutional layers followed by max-pooling, dropout, and fully connected layers.
3. Training: The model is trained using the provided training dataset. Training progress and metrics are displayed during training.
4. Evaluation: After training, the model is evaluated on a separate test dataset. Performance metrics such as loss, precision, and recall are plotted for visualization.
5. Prediction: The trained model is used to make predictions on audio clips. The code processes each audio clip, generates predictions, and groups consecutive detections.
6. Saving Results: The results of the predictions are saved in a CSV file named "results.csv". Each row contains the filename of the audio clip and the number of predicted Capuchin bird calls.


## Project 3: Sentiment Analysis using LSTM and GloVe Embeddings

### Description 
This project demonstrate how to build a deep learning model to predict emojis based on input text using Long-Short Term Memory (LSTM) networks and pre-trained GloVe word embeddings.

### Requirements

-Install Required Libraries

    pip install emoji keras numpy pandas

### Usage

1. Dataset: The dataset used for this project ("emoji_data.csv") contains text samples paired with the emoji labels. Each row consists of a text sample and its correspponding emoji label.
2. Embeddings: The project uses pre-trained GloVe word embeddings to represent words in the input text. These embeddings capture the semantic relationships between words, which will help the model understand the context of the texts.
3. Model Architecture: The model architecture consists of an embedding layer followed by LSTM layers. The embedding layer maps words to dense vectors using pre-trained GloVe embeddings.
4. Training: The model is trained on the provided dataset using categorical cross-entropy and the Adam optimizer.
5. Testing: After training, the model is tested on new text samples to predict the emoji.
