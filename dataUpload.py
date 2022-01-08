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

gfile = drive.CreateFile({'parents': [{'id': 'folderLocation'}]})
# Read file and set it as the content of this instance.
gfile.SetContentFile(sys.argv[1])
gfile.Upload() # Upload the file.
subprocess.Popen(["rm", sys.argv[1]])
print("Uploaded " + sys.argv[1])
