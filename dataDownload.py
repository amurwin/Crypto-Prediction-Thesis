from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import sys
import subprocess

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt") # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)
file_list = drive.ListFile({'q': "'1gwh3hSm3R--5en_2KrqrT9ksOofRTw5N' in parents and trashed=false"}).GetList()
print([file1['title'] for file1 in file_list])
for file1 in file_list:
    file1.GetContentString() # Gets content as string
