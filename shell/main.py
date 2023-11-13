import argparse
from train import *
from prediction import *
from models.autoregressive import AutoRegressiveModel
from models.lstm import LSTMModel



if __name__ == '__main__':
    # Define the argument
    parser = argparse.ArgumentParser(description='Future Data Prediction', 
                                     usage='use "%(prog)s --help" for more information',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', '--name', required=True, 
                        help='''Please type the model name that you would like to use. \n 
                        Type `autoreg` to use Autoregressive model. \n
                        Type `LSTM` to use LSTM deep learning model. \n
                        ''')
    parser.add_argument('-m', '--mode', required=True, 
                        help='''Please type the model you would like to use. \n
                        Type `train` to train the model. \n
                        Type `predict` to make predictions. \n
                        ''')
    parser.add_argument('-p', '--path', required=True, 
                        help='Please provide the relative path to the CSV file')
    args = parser.parse_args()

    # Parse the argument
    csv_file = args.path
    model = None

    # Monitor the skew
    default_dataset_path = '../data_daily.csv'
    check_skew = skew_detection(default_dataset_path, csv_file)
    if check_skew:
        print("Warning: the uploaded dataset has skew with the original training dataset. Please retrain the model.")
    else: 
        print("Dataset is normal.")

    # Train models
    if args.mode == 'train':
        if args.name == 'autoreg':
            model = AutoRegressiveModel()
            train_ar(model, csv_file)
        elif args.name == 'lstm':
            # Hyperparameters
            input_dim = 1
            hidden_dim = 32
            num_layers = 2
            output_dim = 1

            # Instantiate the model
            model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
            train_lstm(model, csv_file)

    # Make predictions
    elif args.mode == 'predict':
        if args.name == 'autoreg':
            prediction = prediction_ar(csv_file, 3)
            print(f"The prediction results for future year's monthly total number is {prediction}")
        elif args.name == 'lstm':
            # Hyperparameters
            input_dim = 1
            hidden_dim = 32
            num_layers = 2
            output_dim = 1

            # Instantiate the model
            prediction = prediction_lstm(csv_file)
            print(f"The prediction results for future year's monthly total number is {prediction}")