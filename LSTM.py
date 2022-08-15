import torch
import torch.nn as nn

class LSTM(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size=columnCount, hidden_layer_size=256, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers = 1).to(device)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        model.hidden_cell = (torch.zeros(1, input_seq.shape[1], model.hidden_layer_size).to(device),
            torch.zeros(1, input_seq.shape[1], model.hidden_layer_size).to(device))
        
        #print("input shape: ", input_seq.shape)
        lstm_out, self.hidden_cell = self.lstm( input_seq.reshape(len(input_seq), -1, columnCount), self.hidden_cell )
        #print("lstm_out shape: ", lstm_out.shape)
        last_output_lstm = lstm_out[-1]

        #print("last out", last_output_lstm.shape)
        predictions = self.linear(last_output_lstm)
        #print("preds shape: ", predictions.shape)

        return predictions