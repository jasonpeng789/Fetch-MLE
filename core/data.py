import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_csv_file(filepath:str) -> pd.DataFrame:
    # Load the csv file and build a dataframe
    dataframe = pd.read_csv(filepath)
    return dataframe


def data_process(filepath:str, visual=False):
    # Get the loaded dataframe
    df = load_csv_file(filepath)

    # Transform the '# Date' data to pandas DateTime format
    df['# Date'] = pd.to_datetime(df['# Date'])

    # Set the '# Date' column as the index of the DataFrame
    df.set_index('# Date', inplace=True)

    # Get the monthly total numbers by aggregate the data
    df_monthly = df.resample('M').sum()

    # Handle the 'nan' value if there is one in the dataset
    df_monthly.fillna(0, inplace=True)

    # Visualize the processed data
    if visual:
        # Plot the monthly receipt counts to visualize trends and seasonality
        plt.figure(figsize=(14, 7))
        plt.plot(df_monthly.index, df_monthly['Receipt_Count'], marker='o')
        plt.title('Monthly Receipt Counts for 2021')
        plt.xlabel('Month')
        plt.ylabel('Total Receipt Count')
        plt.grid(True)

        # Save the data trend graphs
        data_trend_saving_path = 'graphs/data.png'
        plt.savefig(data_trend_saving_path, dpi=300) 
        
        plt.show()

    return df_monthly


def create_sequences(data: np.array, seq_length: int = 3) -> tuple:
    """
    Creates sequences and targets from the provided data for time series prediction.

    Parameters:
    - data (np.array): The time series data.
    - seq_length (int, optional): The length of the sequences. Default is 3.

    Returns:
    - tuple: A tuple containing sequences and targets as numpy arrays.
    """
    sequences = []
    target = []

    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length] if isinstance(data, pd.Series) else data[i:i+seq_length]
        label = data.iloc[i+seq_length] if isinstance(data, pd.Series) else data[i+seq_length]
        sequences.append(seq)
        target.append(label)

    return np.array(sequences), np.array(target)


# Function to normalize data
def normalize(data, mean, std):
    return (data - mean) / std


# Function to denormalize data
def denormalize(data, mean, std):
    return data * std + mean


def data_split(X: np.array, y: np.array, dl=False,  split_rate: float = 0.8) -> tuple:
    """
    Splits the data into training and validation sets and converts them to PyTorch tensors.

    Parameters:
    - X (np.array): The sequences.
    - y (np.array): The targets.
    - split_rate (float, optional): The proportion of data to be used for training. Default is 0.8.

    Returns:
    - tuple: A tuple containing training and validation data as PyTorch tensors.
    """
    # Split the dataset into train and valiation
    train_size = int(split_rate * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    if dl:
        # Convert the datasets to PyTorch tensor
        X_train = torch.FloatTensor(X_train).unsqueeze(2)   # Adding channel dimension
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val).unsqueeze(2)       # Adding channel dimension
        y_val = torch.FloatTensor(y_val)

    return X_train, y_train, X_val, y_val