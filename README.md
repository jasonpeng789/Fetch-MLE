# Fetch Rewards Machine Learning Engineer Exercise
## Introduction
At fetch, we are monitoring the number of the scanned receipts to our app on a daily base as one of our KPIs. From business standpoint, we sometimes need to predict the possible number of the scanned receipts for a given future month.

In this project, we provide the number of the observed scanned receipts each day for the year 2021. Based on this prior knowledge, please develop an algorithm which can predict the approximate number of the scanned receipts for each month of 2022.

## Models Selection
For this project, I provided two different models to perform predictions: Autoregressive Model and LSTM model.
-   Autoregressive Model is a type of statistical model used for understanding and predicting future values in a time series. It regresses the variable against itself (its own lagged values).
-  LSTM Model is a special kind of Recurrent Neural Network (RNN), specifically designed to address the problem of long-term dependencies that traditional RNNs struggle with. The core idea behind LSTMs is the ability to retain information for long periods of time within the network's memory.

## Pipeline Design
For this project, the pipeline of processing the data, training and evaluating models, predicting the results, and monitoring the system, is designed as the following chart:
![image](pipeline.png)

## How to use?

## What's next?

