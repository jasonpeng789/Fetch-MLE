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
### Run the pipeline via docker file
1. Please pull the docker image from the [docker image](https://hub.docker.com/r/jasonpeng789/fetch-ml-app) or run the following command in Terminal/Shell:
```sh
$ docker pull jasonpeng789/fetch-ml-app
```

2. I have provided the `docker-compose.yml`. Hence, please run the folloing command to initizlize the pipeline
```sh
$ docker-compose up
```

3. Connect to the website:
```
localhost:8000
```
**Note** Please note that I set the portal as **8000**. If you would like to use different portal, please modify `ports` from `docker-compose.yml`

4. Click `choose file` to upload your 2021 csv file, and then select the model you would like to have a try. The web app will use this CSV file to predict numbers for each months in 2022. Then click `Upload and Predict` to get the prediction results. 

5. Disconnect the website and stop the docker image:
```sh
$ ctrl + c
$ docker-compose down
```

### Run the pipeline on Terminal/Shell
I have provided the modified code in the `shell` folder if you would like to run the pipeline on your Terminal/Shell

1. Install the requirement pacakges
```sh
$ pip install -r requirements.txt
```

2. Go to the `shell` folder
```sh
$ cd shell
```

3. Run the `main.py`. Please note that you are required to input the following paramaters:

- -n: Represent the name of the model. Type `autoreg` to use Autoregressive model, or `lstm` to use LSTM model
- -m: Represent the mode of the pipeline. Type `train` to train the model, or `predict` to use the model to make predictions
- -p: Represent the path to the CSV file. Please use the relative path. For example, `../data_daily.csv`

Here are some example commands:
- To train an autoreg model
```sh
$ python3 main.py -n autoreg -m train -p ../data_daily.csv
```
- To use the trained autoreg model to make predictions
```sh
$ python3 main.py -n autoreg -m predict -p ../data_daily.csv
```
- To train a LSTM model
```sh
$ python3 main.py -n lstm -m train -p ../data_daily.csv
```
- To use the trained LSTM model to make predictions
```sh
$ python3 main.py -n lstm -m predict -p ../data_daily.csv
```
## What's next?
This pipeline is a simple version. If we can have more resources in the future, here is a list that we can improve the pipeline

- **Re-train the model** Currently we only have 1 year data and the dataset only contains 12 monthly total numbers. This mean the size of the training dataset is extremely small, which can affect the performance of the models. We should use much larger dataset to retrain the models.

- **Experiments with more models** Due to the time and resources limitations, I only provide two models for this pipeline. However, there are more options that may provide better solutions for this pipeline. We should do some more experiments to select the most optimal one.

- **Monitor the pipeline** For this pipeline I only implemented a monitor function to check if the user upload a dataset that has big skew with the dataset that we used to train the models. However, we should add more monitor functions to monitor the pipeline in case it crash or model decay.

- **MLOps** We should implement automatic CI/CD/CT pipeline to maintain the life cycle of the model deployment by using some cloud resouces.

- **UI/UX** I only provide a very simple web UI/UX functions for this project. If possible, we should improve the UI/UX of the web and provide better experiences for the users.

