import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1
sns.get_dataset_names()
flight_data = sns.load_dataset("flights")
flight_data.head()

# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 15
# fig_size[1] = 5
# plt.rcParams["figure.figsize"] = fig_size

# plt.title('Month vs Passenger')
# plt.ylabel('Total Passengers')
# plt.xlabel('Months')
# plt.grid(True)
# plt.autoscale(axis='x',tight=True)
# plt.plot(flight_data['passengers'])

# 2
all_data = flight_data['passengers'].values.astype(float)

# 3
test_data_size = 12
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

# 4
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

# 5
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1).to(device)

# 6
train_window = 12
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

# 7
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size).to(device)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(device),
                            torch.zeros(1,1,self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Learn rate will likely be in [0.0001, 0.01]


training_loss = []

# 8
epochs = 150
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
        training_loss.append(single_loss.item())
        # print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# 9
fut_pred = 12

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

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)




figure, axis = plt.subplots(2,1)

x = np.arange(132, 144, 1)
axis[0].set_title('Month vs Passenger')
axis[0].set_ylabel('Total Passengers')
axis[0].grid(True)
axis[0].autoscale(axis='x', tight=True)
axis[0].plot(flight_data['passengers'])
axis[0].plot(x,actual_predictions)
plt.show()