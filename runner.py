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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_window = 50

monthDF = pd.read_csv("TestData/DOGE_JAN2022.csv", names=['ask_price', 'bid_price',
                 'mark_price', 'high_price', 'low_price', 'open_price', 'volume', 'Time'])
workingdate = datetime(2022, 1, 2)
validDF = True
while validDF: # A full 72 hours was retrieved
    for interval in [10, 300, 3600]: # 10 seconds, 5 minutes, 1 hour
        df = helper.getData(monthDF, workingdate, 72, interval)
        
        # If unable to retrieve 72 Hours, break the loop
        if type(df) == bool:
            validDF = False
            break

        df = df.reset_index()
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
        predSet = x[-1][1:].tolist()
        predSet.append(model(x[-1]).item())
        predSet = torch.FloatTensor(predSet)
        
        for _ in range(0, 1000):
            predSet = predSet[1:].tolist() #List




        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Time"], y=df["ask_price"], line_shape='linear'))
        fig.add_trace(go.Scatter(x=[datetime.strptime(df['Time'][i], "%Y-%m-%d %H:%M:%S.%f") for i in range(len(df)-len(preds), len(df))], y=preds, line_shape='linear'))
        fig.show()
        validDF = False

        torch.save(model.state_dict(), "10000test" + str(interval) + ".lr")