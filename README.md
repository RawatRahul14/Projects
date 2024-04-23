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

Install Required Libraries
   
    !pip install tensorflow tensorflow-io

Mount Google Drive
   
    from google.colab import drive
    drive.mount("/content/drive")

Extrtact Audio Data
   
    with zipfile.ZipFile('/content/drive/MyDrive/Audio_Classification_Data/Data.zip') as z:
        z.extractall()

Python Dependencies

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
