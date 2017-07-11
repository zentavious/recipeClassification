# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:28:03 2017

@author: mzent
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("trainingData4.csv", encoding="latin-1")
test = pd.read_csv("testData4.csv", encoding="latin-1")

cols = ['chinese','dark_soy','peanut_oil','bean_sprout','ginger_root','skim_milk','curd','light_soy','nutmeg','msg','salt','onion','butter','sugar','soy_sauce','ginger','rice','oil','mushrooms','garlic','tofu']
colsRes = ['Category']

trainArr = train.as_matrix(cols)
trainRes = train.as_matrix(colsRes)

rf = RandomForestClassifier(n_estimators=10)
rf.fit(trainArr, trainRes)

testArr = test.as_matrix(cols)
results = rf.predict(testArr)

test['predictions'] = results

#print(test)
correct = 0
total = 0
for i in range(0,len(test['predictions'])):
    total+=1
    if(test['predictions'][i] == test['Category'][i]):
        correct +=1
print(str(correct) +"/"+str(total))


"""
726/1000 Data3
835/1000 Data4

"""