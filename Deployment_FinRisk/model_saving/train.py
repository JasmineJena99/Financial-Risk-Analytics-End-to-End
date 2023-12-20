import joblib
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#load the data
df = pd.read_csv('Default.csv')
#print(df.columns)


## treating outliers present in the 'balance' variable
Q1,Q3 = df['balance'].quantile([.25,.75])
IQR = Q3 -Q1

#LL:lower limit
#UL: upper limit

LL= Q1 - 1.5*IQR
UL= Q3 + 1.5*IQR

#convert all the values which are above UL  to the UL ; i.e if UL = 2193, and outliers = 2838 or 2561, they will convert to 2193
df['balance'] = np.where(df['balance']>UL,UL,df['balance'])



#encoding
df = pd.get_dummies(df, drop_first=True)

#relabeling the columns as per original names
df.columns = ['balance', 'income', 'default','student']


#separating dependent and independent variables
x = df.drop('default', axis =1)
y = df['default']

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state =101)

#treating target imbalance using SMOTE
sm = SMOTE(random_state =33, sampling_strategy = 0.75)
x_res,y_res = sm.fit_resample(x_train, y_train)

train_smote = pd.concat([x_res,y_res], axis=1)
test = pd.concat([x_test,y_test], axis=1)

#train model
model = LogisticRegression()
model.fit(x_res,y_res)

print("[INFO] model trained")

score = model.score(x_test,y_test) #this line will predict y for all x_test and give accuracy
print(score)

#saving of the model using joblib

try:
    joblib.dump(model, 'finrisk_joblib.pkl')
    print("Model saved successfully.")
except Exception as e:
    print("An error occurred:", e)
