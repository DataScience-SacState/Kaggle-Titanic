import pandas as pd
import numpy as np
import csv as csv

#read the train set into pandas
train_df = pd.read_csv('data/train.csv', header=0) 

train = train_df

print( sum(train["Survived"]) / float(len(train["Survived"])) )
#print(train_df)
