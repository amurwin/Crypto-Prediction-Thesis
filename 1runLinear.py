from datetime import datetime
from sqlalchemy import false
import helper
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from dataGen import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_window = 50
startWindow = 25920 #Day 4 12:00AM

monthDF = pd.read_csv("TestData/DOGE_JAN2022.csv", names=['ask_price', 'bid_price',
                 'mark_price', 'high_price', 'low_price', 'open_price', 'volume', 'Time'])
monthFilters = [pd.read_pickle("10s.pkl"), pd.read_pickle("60s.pkl"), pd.read_pickle("3600s.pkl")]
monthFilters = [x.reset_index(drop=True) for x in monthFilters]
frameGen = dataGenerator(monthDF, datetime(2022, 1, 1))

allDFs = next(frameGen)
allDFs = allDFs.reset_index(drop=True)
filteredDFs = dataFilter(allDFs, [timedelta(seconds=10), timedelta(seconds=60), timedelta(seconds=3600)])
df = filteredDFs[0]
df = df.reset_index(drop=True)

train_df = df[['ask_price']].values.astype(float)


# LinearRegression
x, y = helper.ls2(train_df, train_window)
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
train_ds = TensorDataset(x, y)
train_dl = DataLoader(train_ds)

# Define linear model
model = nn.Linear(train_window, 1)

# Parameters
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=0.001)
loss = 0 # Pre-initialized for scoping access

#fit model for 1000 epochs
helper.fit(1000, model, loss_fn , opt ,train_dl)

# Generate predictions
preds = model(x)
preds = preds.reshape(-1).tolist()
newPreds = []

#Initial Pred Set
predSet = x[-1][1:].tolist()
predSet.append(model(x[-1]).item())
predSet = torch.FloatTensor(predSet)
newPreds.append(model(predSet).item())

thisTime = startWindow + 1

while (max(newPreds[-1], monthFilters[0]["ask_price"].iloc[thisTime]) / min(newPreds[-1], monthFilters[0]["ask_price"].iloc[thisTime])) < 1.03:
    predSet = predSet[1:].tolist() #List
    predSet.append(newPreds[-1])
    predSet = torch.FloatTensor(predSet)
    newPreds.append(model(predSet).item())
    thisTime += 1


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["Time"], y=df["ask_price"], line_shape='linear'))
fig2.add_trace(go.Scatter(x=[datetime.strptime(df['Time'][i], "%Y-%m-%d %H:%M:%S.%f") for i in range(len(df)-len(preds), len(df))], y=preds, line_shape='linear'))
fig2.add_trace(go.Scatter(x=[datetime(2022, 1, 1) + timedelta(seconds = 10 * x) for x in range(startWindow,startWindow + len(newPreds))],y=newPreds,line_shape='linear'))
fig2.add_trace(go.Scatter(x=monthFilters[0][startWindow:thisTime]["Time"], y=monthFilters[0][startWindow:thisTime]["ask_price"], line_shape='linear'))
fig2.show()

startWindow += 360
torch.save(model.state_dict(), "10000test" + str(11) + ".lr")