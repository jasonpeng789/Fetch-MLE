import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A PyTorch implementation of an LSTM model for time series prediction.

    Attributes:
    - hidden_dim (int): The number of hidden units in the LSTM layers.
    - num_layers (int): The number of LSTM layers.
    - lstm (nn.LSTM): The LSTM layers.
    - fc (nn.Linear): The fully connected layer.

    Methods:
    - forward(x): Performs a forward pass through the model.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        """
        Initializes the LSTMModel.

        Parameters:
        - input_dim (int): The number of input features.
        - hidden_dim (int): The number of hidden units in the LSTM layers.
        - num_layers (int): The number of LSTM layers.
        - output_dim (int): The number of output features.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor.
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

