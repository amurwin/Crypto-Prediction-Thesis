import time
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1
df = pd.read_csv("TestData/file.csv", names=['ask_price', 'bid_price',
                 'mark_price', 'high_price', 'low_price', 'open_price', 'volume', 'Time'])
df = df.iloc[::500]



# 2
print('2')
all_data = df['ask_price'].values.astype(float)

# 3
print('3')
test_data_size = int(len(df.index) * 0.2)
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

# 4
print('4')
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

# 5
print('5')
train_data_normalized = torch.FloatTensor(
    train_data_normalized).view(-1).to(device)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 6
print('6')
for train_window in [110, 120]:

    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    # 7
    print('7')


    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size).to(device)

            self.linear = nn.Linear(hidden_layer_size, output_size)

            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                                torch.zeros(1, 1, self.hidden_layer_size).to(device))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(
                input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    model = LSTM().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Learn rate will likely be in [0.0001, 0.01]

    start = time.time()

    training_loss = []

    # 8
    print('8')
    epochs = 1500
    for i in range(epochs):
        j = 0
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                          torch.zeros(1, 1, model.hidden_layer_size).to(device))
            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            training_loss.append(single_loss.item())
            if j % 1000 == 0:
                print(f'epoch: {i:3} seq: {j:3}')
            j += 1
            # print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    end = time.time()
    print('Time Elapsed')
    print(end-start)

    # 9
    print('9')
    fut_pred = test_data_size

    test_inputs = train_data_normalized[-train_window:].tolist()
    print(test_inputs)

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).to(device))
            test_inputs.append(model(seq).item())

    print(test_inputs[fut_pred:])

    actual_predictions = scaler.inverse_transform(
        np.array(test_inputs[train_window:]).reshape(-1, 1))
    print(actual_predictions)


    figure, axis = plt.subplots(2, 1)

    axis[0].grid(True)
    axis[0].autoscale(axis='x', tight=True)
    axis[0].plot(df['Time'][-(fut_pred + 100):],
                df['ask_price'][-(fut_pred + 100):])
    axis[0].plot(df['Time'][-fut_pred:], actual_predictions)
    axis[1].grid(True)
    axis[1].autoscale(axis='x', tight=True)
    axis[1].plot(df['Time'], df['ask_price'])
    axis[1].plot(df['Time'][-fut_pred:], actual_predictions)
    plt.show()
