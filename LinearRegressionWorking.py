from datetime import datetime
import helper
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from dataGen import *

train_window = 50
start_day = 1
start_hour = 0
timeframe = 60  # Number of seconds 
day_duration = 86400 / timeframe
hour_duration = day_duration / 24
prerun_model = '5000-0001-R2-F1.linear' # Optional

if not hour_duration.is_integer():
    raise Exception("INVALID TIMEFRAME")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
monthDF = pd.read_csv("TestData/DOGE_JAN2022.csv", names=['ask_price', 'bid_price',
                 'mark_price', 'high_price', 'low_price', 'open_price', 'volume', 'Time'])
monthFilters = [pd.read_pickle("10s.pkl"), pd.read_pickle("60s.pkl"), pd.read_pickle("3600s.pkl")]
monthFilters = [x.reset_index(drop=True) for x in monthFilters]
train_df = monthFilters[1].iloc[int(day_duration*(start_day-1)+hour_duration*start_hour):int(day_duration*(start_day+2)+hour_duration*start_hour)].copy(deep=True).reset_index(drop=True)
test_df = monthFilters[1].iloc[int(day_duration*(start_day+2)+hour_duration*start_hour):int(day_duration*(start_day+3)+hour_duration*start_hour)].copy(deep=True).reset_index(drop=True)

train_df_ask_price = train_df[['ask_price']].values.astype(float)
test_df_ask_price = test_df[['ask_price']].values.astype(float)

train_x, train_y = helper.ls2(train_df_ask_price, train_window)
test_x, test_y = helper.ls2(test_df_ask_price, train_window)

train_x_Tensor = torch.FloatTensor(train_x)
train_y_Tensor = torch.FloatTensor(train_y)

test_x_Tensor = torch.FloatTensor(test_x)
test_y_Tensor = torch.FloatTensor(test_y)

train_dataset = TensorDataset(train_x_Tensor, train_y_Tensor)
train_dataloader = DataLoader(train_dataset, shuffle=True)

model = nn.Linear(train_window, 1)


if True: # If false, load a pre-run model
    loss_fn = F.mse_loss
    opt = torch.optim.SGD(model.parameters(), lr=0.0005)
    loss = 0
    helper.fit(10000, model, loss_fn , opt ,train_dataloader)
    torch.save(model.state_dict(), '10000-0005-T1.linear')
else:
    model.load_state_dict(torch.load(prerun_model))

# Autoregressive Predictions
newPreds = torch.Tensor([])
predSet = torch.tensor([x[0] for x in test_df_ask_price[0:50].tolist()])

newPreds = torch.cat([newPreds, model(predSet)])
predSet = torch.cat([predSet[1:], newPreds[-1].reshape(-1)])
index = 1

while index + 50 < len(test_df) and max(newPreds[-1], test_df['ask_price'].iloc[index + 50])/min(newPreds[-1], test_df['ask_price'].iloc[index + 50]) < 1.03:
    newPreds = torch.cat([newPreds, model(predSet)])
    predSet = torch.cat([predSet[1:], newPreds[-1].reshape(-1)])
    index += 1

# Generate Graph
graph = go.Figure()
graph.add_trace(go.Scatter(x=train_df["Time"], y=train_df["ask_price"], line_shape="linear", name="Training data"))
graph.add_trace(go.Scatter(x=train_df['Time'].iloc[train_window:], y=model(train_x_Tensor).reshape(-1).tolist(), line_shape="linear", name="Model on Training data"))
graph.add_trace(go.Scatter(x=test_df["Time"], y=test_df["ask_price"], line_shape="linear", name="Test data"))
graph.add_trace(go.Scatter(x=test_df['Time'].iloc[train_window:], y=model(test_x_Tensor).reshape(-1).tolist(), line_shape="linear", name="Model on Test data"))
graph.add_trace(go.Scatter(x=test_df['Time'].iloc[train_window:train_window+len(newPreds)], y=newPreds.tolist(), line_shape="linear", name="Autoregressive Prediction"))
graph.show()