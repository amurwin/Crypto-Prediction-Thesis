from datetime import datetime, timedelta

currDate = datetime(2022, 1, 1)
jointfile = open("Data/DOGE_JAN2022.csv", "w+")

while currDate < datetime(2022, 1, 31):
    try:
        with open("Data/DOGE_" + currDate.strftime("%m%d%Y") + ".csv") as file:
            for line in file:
                jointfile.write(line)
    except:
        print( "No data on " + currDate.strftime("%m/%d/%Y"))

    currDate += timedelta(days=1)