from copy import deepcopy
from datetime import datetime, timedelta
import pandas as pd

def getDataDF(a, b, c, d):
    return getData(a, b, c, d)
    

def getData(df, startTime=datetime(1900, 1, 1), duration=72, timeStep=1):
    workingTime = startTime

    ndf = pd.DataFrame(columns=df.columns)
    startIndex = 0

    for i in range(0, len(df)):
        if datetime.strptime(df['Time'][i], "%Y-%m-%d %H:%M:%S.%f") > startTime:
            startIndex = i
            break

    # Check all possible data
    for i in range(startIndex, len(df)):
        # Get the time of the datapoint
        dpt = datetime.strptime(df['Time'][i], "%Y-%m-%d %H:%M:%S.%f")

        # If its over 72 hours, break out
        if dpt > startTime + timedelta(hours=duration):
            return ndf

        # If the next datapoint in valid, add it to the list and update
        # when the next valid data time is
        if dpt > workingTime:
            ndf = pd.concat([ndf, pd.DataFrame(df.iloc[i]).T])
            workingTime += timedelta(seconds=timeStep)
    return False
    

def create_linear_sequences(input_data, tw):
    input_seq = []
    output_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw]
        input_seq.append(train_seq)
        output_seq.append(train_label)
    return input_seq, output_seq

def ls2(all_data, train_window):
    input_seq = []
    output_seq = []
    L = len(all_data)
    for i in range(L-train_window):
        train_seq = [v[0] for v in all_data[i:i+train_window]]
        train_label = all_data[i+train_window]
        input_seq.append(train_seq)
        output_seq.append(train_label)
    return input_seq, output_seq

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    losses = []
    loss = 0
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            opt.zero_grad()
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
        
            #Update loss list for each epoch
            losses.append(loss)

        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


    return losses