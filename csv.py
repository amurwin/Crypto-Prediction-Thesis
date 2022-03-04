from datetime import datetime, timedelta

currDate = datetime(2021, 8, 1)
jointfile = open("Data/out.csv", "w+")

while currDate < datetime(2022, 2, 15):
    try:
        with open("Data/DOGE_" + currDate.strftime("%m%d%Y") + ".csv") as file:
            for line in file:
                jointfile.write(line)
    except:
        print( "No data on " + currDate.strftime("%m/%d/%Y"))

    currDate += timedelta(days=1)