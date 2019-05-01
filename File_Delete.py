import os

os.chdir('')

#permanent delete
for filename in os.listdir():
    if filename.endwith('.txt'):
        os.unlink(filename)

#move to trash:
import send2trash




os.unlink() will delete a file
os.rmdir() will delete a folder (but the folder must be empty)
shutil.rmtree() will delete a folder and all its contents
send2trash.send2trash() will send a file or folder to the recycling bin
