import pickle
from data import *
from core.models.autoregressive import AutoRegressiveModel
from core.models.lstm import LSTMModel


def prediction_ar(csv_file, seq_length=3):
    # Initialize the model
    model = AutoRegressiveModel()

    # Load and processed the dataset
    df_monthly = data_process(csv_file, visual=False)

    # Calculate mean and standard deviation for normalization
    mean = df_monthly['Receipt_Count'].mean()
    std = df_monthly['Receipt_Count'].std()

    # Normalize the latest data to use as the base for prediction
    latest_data = df_monthly['Receipt_Count'][-seq_length:].values
    normalized_data = normalize(latest_data, mean, std)

    # Load the saved model
    with open('core/trained_models/ar_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Predict for each month of 2022
    predictions = []
    for _ in range(12):  # 12 months
        # Predict the next value
        normalized_prediction = model.predict(normalized_data.reshape(1, -1))[0]
        # Denormalize the prediction
        prediction = denormalize(normalized_prediction, mean, std)
        predictions.append(int(prediction))
        
        # Update the sequence for the next prediction
        normalized_data = np.roll(normalized_data, -1)
        normalized_data[-1] = normalize(prediction, mean, std)

    return predictions


def prediction_lstm(csv_file):
    # Initialize the model
    # Hyperparameters
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    # Instantiate the model
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

    # Load and processed the dataset
    df_monthly = data_process(csv_file, visual=False)

    # Calculate mean and standard deviation for normalization
    mean = df_monthly['Receipt_Count'].mean()
    std = df_monthly['Receipt_Count'].std()

    # Normalize the latest data to use as the base for prediction
    input = df_monthly['Receipt_Count']

    # Load the trained paramaters
    model_path = 'core/trained_models/lstm_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Store the number of each months of 2022
    predictions = []

    # Prediction loop
    for month in range(12):
        # Normalize the data
        mean = np.mean(input)
        std = np.std(input)
        input_normalized = (input - mean) / std
        input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).unsqueeze(2)

        # Prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Denormalize
        predicted_value = (prediction.numpy() * std) + mean
        predictions.append(predicted_value[0][0])
   
        # Update the prediction dataset
        input = np.append(input, predicted_value[0][0])[1:]

    #print(f"Predicted number of scanned receipts for each month of 2022: {predictions}")
    return predictions
