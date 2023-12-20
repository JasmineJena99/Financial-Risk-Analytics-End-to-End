from flask import Flask, render_template, request
import joblib

# initiate the app

app = Flask(__name__)


#load the model
model = joblib.load('finrisk_joblib.pkl') #the joblib file should be in the same directory as main.py

# @app.route('/') #its a decorator; if someone hits '/', this following function will run automatically
# def home():
#     return 'hello world'
#
#
# @app.route('/contact-us')
# def contact():
#     return 'Welcome to contact us page'

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['post'])
def predict():
    balance = request.form.get('balance')
    income = request.form.get('income')
    student = request.form.get('student')



    print(balance,income,student) #to check if the value is returned to the back end

    output = model.predict([[int(balance),int(income),int(student)]])

    if output[0]==True:
        return "Person is defaulter"
    return 'Person Not defaulter'


'''
@app.route('/learnbay')
def learnbay():
    return render_template('learnbay.html')
'''

# run the app
app.run()