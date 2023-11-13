import pickle
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from data import *
from models.autoregressive import AutoRegressiveModel
from models.lstm import LSTMModel


def train_ar(model: AutoRegressiveModel, csv_file: str):
    """
    Trains an autoregressive model using the provided dataset.

    The function processes the dataset from a CSV file, normalizes the data, creates sequences, 
    and splits them into training and validation sets. The model is then trained on the training set 
    and evaluated on the validation set. The trained model is saved to disk.

    Parameters:
    model (AutoRegressiveModel): The autoregressive model instance to be trained.
    csv_file (str): The path to the CSV file containing the data.

    Returns:
    None: This function does not return a value but prints the model's mean squared error (MSE) 
          after training and saves the model to disk.
    """
    # Load and processed the dataset
    df_monthly = data_process(csv_file, visual=False)

    # Data normalization 
    mean = df_monthly['Receipt_Count'].mean()
    std = df_monthly['Receipt_Count'].std()
    df_monthly['Receipt_Count'] = (df_monthly['Receipt_Count'] - mean) / std

    X, y = create_sequences(df_monthly['Receipt_Count'], seq_length=3)
    X_train, y_train, X_val, y_val = data_split(X, y, split_rate=0.8)
    
    model = AutoRegressiveModel()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    mse = np.mean((predictions - y_val) ** 2)
    print(f"Autoreg Model MSE: {mse:.2f}")

    # Save the trained model
    with open('trained_models/ar_model.pkl', 'wb') as file:
        pickle.dump(model, file)


def train_lstm(model: LSTMModel, csv_file: str):
    """
    Trains a Long Short-Term Memory (LSTM) model using the provided dataset.

    The function processes the dataset from a CSV file, normalizes the data, creates sequences, 
    and splits them into training and validation sets. The model is trained over several epochs 
    and evaluated on the validation set. Training and validation losses are plotted and saved as 
    an image, and the trained model is saved to disk.

    Parameters:
    model (LSTMModel): The LSTM model instance to be trained.
    csv_file (str): The path to the CSV file containing the data.

    Returns:
    None: This function does not return a value but prints training and validation losses, saves 
          the loss plot, and saves the trained model to disk.
    """
    # Load and processed the dataset
    df_monthly = data_process(csv_file, visual=False)

    # Data normalization 
    mean = df_monthly['Receipt_Count'].mean()
    std = df_monthly['Receipt_Count'].std()
    df_monthly['Receipt_Count'] = (df_monthly['Receipt_Count'] - mean) / std

    X, y = create_sequences(df_monthly['Receipt_Count'], seq_length=3)
    X_train, y_train, X_val, y_val = data_split(X, y, dl=True, split_rate=0.8)

    # Load the dataset
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=4)

    # Hyperparameters
    learning_rate = 0.001
    epochs = 200

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store losses
    train_losses = []
    val_losses = []

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the trained model and evaluation image
    trained_model_name = 'lstm_model.pth'
    model_loss_name = 'lstm_loss_plot.png'
    model_saving_path = 'trained_models/' + trained_model_name
    image_saving_path = 'graphs/' + model_loss_name
    torch.save(model.state_dict(), model_saving_path)
    plt.savefig(image_saving_path, dpi=300) 
    
    plt.show()

