#this file will be responsible ONLY for prediction
import pickle
import joblib


#model = pickle.load(open('dib_79.pkl', 'rb'))
model = joblib.load('finrisk_joblib.pkl')
output = model.predict([[1487.00,17854,1]])
print(output)