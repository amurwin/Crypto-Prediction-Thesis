import time
from sqlalchemy import true
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
df = df.iloc[::]


# 2
print('2')
all_data = df[['ask_price', 'bid_price', 'mark_price']].values.astype(float)

columnCount = len(all_data[0])
batch_size = 100

# 3
print('3')
test_data_size = int(len(df.index) * 0.2)
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

if columnCount > 1:
    train_data = train_data.reshape(-1, columnCount)
else:
    train_data = train_data.reshape(-1)
# Not scaling on normalizing since datapoints are of a similar nature

def create_inout_sequences(input_data, tw):
    input_seq = []
    output_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        input_seq.append(train_seq)
        output_seq.append(train_label)
    return input_seq, output_seq

class LSTM(nn.Module):
    def __init__(self, input_size=columnCount, hidden_layer_size=128, output_size=1):
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


train_window = 256

### 
train_input, train_labels = create_inout_sequences(train_data, train_window)

train_dataset = []
for i in range(0, len(train_input), batch_size):
    batched_input = train_input[i : i + batch_size]
    torch_batched_input = torch.FloatTensor(batched_input).to(device)
    ## TRANSPOSING HERE DREW
    torch_batched_input = torch.transpose(torch_batched_input, 0, 1)

    batched_output = train_labels[i : i + batch_size]
    torch_batched_output = torch.FloatTensor(batched_output).to(device)

    train_dataset.append((torch_batched_input, torch_batched_output))



# train_dataset = train_dataset[len(train_dataset) % batch_size:]


model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Learn rate will likely be in [0.0001, 0.01]

start = time.time()

training_loss = []

# 8
print('8')
epochs = 1000
for i in range(epochs):
    j = 0
    for input, y_real in train_dataset:
        optimizer.zero_grad()

        #print(input.size())
        y_pred = model(input)

        single_loss = loss_function(y_pred, y_real)
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

#test_data = df[['ask_price', 'bid_price', 'mark_price']] for last 20%
#test_data_size = len(test_data)

test_inputs = test_data.tolist()

model.eval()

test_input, test_labels = create_inout_sequences(test_data, train_window)

test_dataset = []
with torch.no_grad():
    for i in range(0, len(test_input)):
        torch_test_input = torch.FloatTensor(test_input[i]).to(device)
        ## TRANSPOSING HERE DREW
        #
        torch_test_input = torch_test_input.reshape(1, train_window, columnCount)
        print(i)
        torch_test_input = torch.transpose(torch_test_input, 0, 1)
        #model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size).to(device),
                        #torch.zeros(1, batch_size, model.hidden_layer_size).to(device))
        test_dataset.append(model(torch_test_input).cpu().numpy()[0][0])

figure, axis = plt.subplots(2, 1)

axis[0].grid(True)
axis[0].autoscale(axis='x', tight=True)
axis[0].plot(df['Time'],df['ask_price'])
axis[0].plot(df['Time'][-(len(test_dataset) + 1):-1], test_dataset)

# axis[1].grid(True)
# axis[1].autoscale(axis='x', tight=True)
# axis[1].plot(df['Time'], df['ask_price'])
# axis[1].plot(df['Time'][-test_data_size:], actual_predictions)
plt.show()
