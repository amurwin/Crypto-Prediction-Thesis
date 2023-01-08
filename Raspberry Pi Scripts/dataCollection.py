import csv
import robin_stocks.robinhood as robin
import pyotp
import time
import sys
import subprocess
from datetime import datetime, timedelta

totp = pyotp.TOTP("twoFactorAuthentication").now()
login = robin.login("emailAddress", "password", mfa_code=totp)
tag = sys.argv[1]
csvName = datetime.now().strftime(tag + "_%m%d%Y.csv")
with open(csvName, 'w', newline='') as output:
    csvWriter = csv.writer(output)
    startTime = datetime.now()
    while datetime.now() < (startTime + timedelta(hours=24)):
        try:
            time.sleep(1)
            data = robin.crypto.get_crypto_quote(tag, info=None)    
            temp = (data['ask_price'], data['bid_price'], data['mark_price'], data['high_price'], data['low_price'], data['open_price'], data['volume'], datetime.now())
            csvWriter.writerow(temp)
        except:
            print("Gateway error")
subprocess.Popen(["python3", "dataUpload.py", csvName, "&"])
