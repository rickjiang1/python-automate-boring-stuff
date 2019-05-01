import nltk
import pandas as pd 
import numpy as np
import os 
#downlaod stopwords package from nltk shell
#nltk.download_shell()

os.chdir('C:\PythonProject\Machine_Learning\dataset')


#.rstrip()  
#The method rstrip() returns a copy of the string in which all chars have been stripped from the end of the string
message=[line.rstrip() for line in open('SMSSpamCollection')]
message=pd.DataFrame(message,columns=['text'])


message['target']=message['text'].apply(lambda x: x.split('\t')[0])
message['text_message']=message['text'].apply(lambda x: x.split('\t')[1])