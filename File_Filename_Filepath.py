import os
'''
os.path.join('folder1','folder2','folder3')
os.getcwd()
os.chdir('')
os.path.abspath()
os.path.dirname('')
os.path.basename('')
os.path.exists('')
os.path.isfile('')
os.path.isdir('')
os.path.getsize('')
os.listdir('')
os.makedirs('')
'''

os.chdir('C:\PythonProject\Machine_Learning')

#get the file size
for i in os.listdir(os.getcwd()):
    print(os.path.getsize(os.path.join(os.getcwd(),i)))


#get the folder1
print('\n')
for i in os.listdir(os.getcwd()):
    print(str(i)+': '+str(os.path.isdir(os.path.join(os.getcwd(),i))))
