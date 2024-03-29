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
