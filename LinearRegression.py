from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import helper
import plotly.express as px
import plotly.graph_objects as go

# 220000 datapoints = ~72 Hours

df = pd.read_csv("TestData/DOGE_JAN2022.csv", names=['ask_price', 'bid_price',
                 'mark_price', 'high_price', 'low_price', 'open_price', 'volume', 'Time'])
ndf = helper.getData(df, datetime(2022, 1, 1), 72, 300)
ndf = ndf.reset_index()

train_window = 50

all_data = ndf[['ask_price', 'bid_price', 'mark_price']].values.astype(float)
x, y = helper.create_linear_sequences(all_data, train_window)

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
train_ds = TensorDataset(x, y)

batch_size = 1
train_dl = DataLoader(train_ds)

# Define linear model
model = nn.Linear(3, 1)
print(model.weight)
print(model.bias)
# Parameters
list(model.parameters())
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=1e-5)
loss = 0
# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
            # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
#fit model for 100 epochs
fit(1000, model, loss_fn , opt ,train_dl)
# Generate predictions
preds = model(x)
preds = preds.reshape(-1).tolist()

fig = go.Figure()
fig.add_trace(go.Scatter(x=ndf["Time"], y=ndf["ask_price"], line_shape='linear'))
fig.add_trace(go.Scatter(x=[datetime.strptime(ndf['Time'][i], "%Y-%m-%d %H:%M:%S.%f") for i in range(0, len(ndf))], y=preds, line_shape='linear'))
fig.show()