import pandas as pd
import numpy as np
import os
os.chdir('C:\PythonProject\Machine_Learning')
EC=pd.read_csv('Ecommerce Purchases.csv')
print(EC.info())

#how many people have a credit card that expires in 2015
print(sum(EC['CC Exp Date'].apply(lambda exp:exp.split('/')[1]=='25')))


#what are the top 5 most popular email providers/hosts
print(EC['Email'].apply(lambda x:x.split('@')[1]).value_counts().head(5))
