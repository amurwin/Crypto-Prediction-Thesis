from datetime import datetime, timedelta
import pandas

def dataGenerator(dataDF: pandas.DataFrame, 
                    startTime: datetime=datetime(2022, 1, 1), 
                    endTime: datetime=datetime(2022, 2, 1),
                    timeframe: timedelta=timedelta(hours=72), 
                    increment: timedelta=timedelta(hours=1)):

    rangeStartTime = startTime
    startIndex = 0
    endIndex = 0

    while(rangeStartTime - increment < endTime - timeframe):
        while(parseTime(dataDF['Time'][startIndex]) < rangeStartTime):
            startIndex += 1

        while(parseTime(dataDF['Time'][endIndex]) < rangeStartTime + timeframe):
            endIndex += 1
        
        rangeStartTime += increment
        yield dataDF.loc[startIndex:endIndex].copy().reset_index(drop=True)

def dataFilter(dataDF: pandas.DataFrame,
                timesteps): #List[timedelta]
    newDFs = [pandas.DataFrame(columns=dataDF.columns) for _ in timesteps]
    nextTimes = [parseTime(dataDF['Time'][0]).replace(second=0, microsecond=0) for _ in timesteps]

    for dfIndex in range(0, len(dataDF)):
        for index, t in enumerate(nextTimes):
            if t < parseTime(dataDF['Time'][dfIndex]):
                newDFs[index] = pandas.concat([newDFs[index], pandas.DataFrame(dataDF.iloc[dfIndex]).T])
                nextTimes[index] += timesteps[index]
        # if dfIndex % 100 == 0:
        #     print("Row: [" + str(dfIndex) + "/" + str(len(dataDF)) + "]")

    return newDFs

def parseTime(string):
    try:
        time = datetime.strptime(string, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        time = datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    return time



def main():
    from datetime import datetime
    import pandas

    monthDF = pandas.read_csv("TestData/DOGE_JAN2022.csv", names=['ask_price', 'bid_price',
                    'mark_price', 'high_price', 'low_price', 'open_price', 'volume', 'Time'])

    gen = dataGenerator(monthDF, datetime(2022, 1, 1))
    x = next(gen)
    print("Filtering Now")
    dataDFs = dataFilter(x, [timedelta(seconds=10), timedelta(seconds=60), timedelta(seconds=3600)])
